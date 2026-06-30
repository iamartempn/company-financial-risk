"""
Streamlit-приложение для скоринга финансовой устойчивости компаний.
"""

import os
import logging

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import polars as pl

logger = logging.getLogger(__name__)

# Путь к базе данных и модели по умолчанию
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw.duckdb")
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lgbm.pkl")

_RISK_COLORS = {
    "Низкий": "#28a745",
    "Средний": "#ffc107",
    "Высокий": "#dc3545",
}


@st.cache_data(show_spinner="Загрузка данных...")
def load_data() -> pl.DataFrame:
    """
    Загружает данные из DuckDB или генерирует синтетические при отсутствии базы.

    Возвращает
    ----------
    pl.DataFrame
        DataFrame с признаками и скорами риска.
    """
    from src.features import compute_ratios
    from src.collector import load_fallback_data

    db_path = _DEFAULT_DB_PATH
    if os.path.exists(db_path):
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        pdf = con.execute("SELECT * FROM raw").df()
        con.close()
        df = pl.from_pandas(pdf)
    else:
        st.warning("База данных не найдена. Используются синтетические данные.")
        df = load_fallback_data()

    df = compute_ratios(df)
    return df


@st.cache_resource(show_spinner="Загрузка модели...")
def load_model_cached():
    """
    Загружает обученную модель из файла.
    При отсутствии файла возвращает None.

    Возвращает
    ----------
    dict или None
        Словарь с моделью и списком признаков.
    """
    from src.model import load_model

    model_path = _DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)


def score_dataframe(df: pl.DataFrame, model) -> pl.DataFrame:
    """
    Применяет модель к DataFrame или присваивает эвристический скор по Z-score.

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с признаками.
    model : dict или None
        Обученная модель или None.

    Возвращает
    ----------
    pl.DataFrame
        DataFrame с добавленными колонками risk_score и risk_label.
    """
    if model is not None:
        from src.model import score
        return score(df, model)

    # Эвристика на основе Z-score при отсутствии модели
    def z_to_risk(z):
        if z is None or np.isnan(z):
            return 0.5
        if z < 1.23:
            return 0.75
        if z < 2.9:
            return 0.45
        return 0.15

    def z_to_label(z):
        if z is None or np.isnan(z):
            return "Средний"
        if z < 1.23:
            return "Высокий"
        if z < 2.9:
            return "Средний"
        return "Низкий"

    zs = df["z_score"].to_list()
    return df.with_columns([
        pl.Series("risk_score", [z_to_risk(z) for z in zs]),
        pl.Series("risk_label", [z_to_label(z) for z in zs]),
    ])


def risk_gauge(risk_score: float, risk_label: str) -> go.Figure:
    """
    Строит gauge-диаграмму для визуализации вероятности дефолта.

    Параметры
    ---------
    risk_score : float
        Вероятность дефолта [0, 1].
    risk_label : str
        Риск-класс.

    Возвращает
    ----------
    go.Figure
        Plotly-фигура с индикатором.
    """
    color = _RISK_COLORS.get(risk_label, "#6c757d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score * 100, 1),
        title={"text": "Вероятность дефолта, %"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 35], "color": "rgba(40,167,69,0.15)"},
                {"range": [35, 65], "color": "rgba(255,193,7,0.15)"},
                {"range": [65, 100], "color": "rgba(220,53,69,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": risk_score * 100,
            },
        },
        number={"suffix": "%", "font": {"size": 32}},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def render_company_card(row: dict, model, df_full: pl.DataFrame) -> None:
    """
    Отображает карточку риска компании с SHAP waterfall.

    Параметры
    ---------
    row : dict
        Словарь с данными одной компании.
    model : dict или None
        Обученная модель.
    df_full : pl.DataFrame
        Полный DataFrame с признаками.
    """
    from src.model import FEATURE_COLS

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### Карточка компании")
        z = row.get("z_score")
        z_str = f"{z:.2f}" if z is not None and not np.isnan(float(z)) else "н/д"
        label = row.get("risk_label", "н/д")
        label_color = _RISK_COLORS.get(label, "#6c757d")

        st.markdown(f"""
| Поле | Значение |
|---|---|
| ИНН | `{row.get('inn', 'н/д')}` |
| Z-score Альтмана | **{z_str}** |
| Риск-класс | <span style='color:{label_color};font-weight:bold'>{label}</span> |
""", unsafe_allow_html=True)

    with col2:
        rs = row.get("risk_score", 0.5)
        rl = row.get("risk_label", "Средний")
        st.plotly_chart(risk_gauge(float(rs), str(rl)), use_container_width=True)

    with st.expander("Вклад признаков (SHAP waterfall)", expanded=True):
        if model is not None:
            from src.explain import compute_shap, waterfall_fig

            # Находим индекс записи в полном DataFrame
            inn_list = df_full["inn"].to_list()
            try:
                idx = inn_list.index(row["inn"])
            except ValueError:
                idx = 0

            feature_cols = model["feature_cols"]
            shap_values, _ = compute_shap(df_full, model)

            fig = waterfall_fig(shap_values, idx, feature_cols)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Модель не загружена. SHAP-объяснения недоступны.")


def render_all_companies_tab(df: pl.DataFrame) -> None:
    """
    Отображает вкладку со scatter-диаграммой по всем компаниям.

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с риск-скорами и Z-score.
    """
    st.markdown("### Все компании: риск-скор vs Z-score")

    if "risk_score" not in df.columns or "z_score" not in df.columns:
        st.info("Недостаточно данных для отображения диаграммы.")
        return

    pdf = df.select(["inn", "z_score", "risk_score", "risk_label"]).to_pandas().dropna(
        subset=["z_score", "risk_score"]
    )

    fig = px.scatter(
        pdf,
        x="z_score",
        y="risk_score",
        color="risk_label",
        color_discrete_map=_RISK_COLORS,
        hover_data=["inn"],
        labels={
            "z_score": "Z-score Альтмана",
            "risk_score": "Вероятность дефолта",
            "risk_label": "Риск-класс",
        },
        title="Распределение компаний по Z-score и вероятности дефолта",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    fig.update_layout(height=480, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


def render_sectors_tab(df: pl.DataFrame) -> None:
    """Отображает распределение риск-классов по отраслям."""
    st.markdown("### Распределение риска по отраслям")

    if "okved_section" not in df.columns:
        # Генерируем условные отрасли из ИНН для демо
        import hashlib
        sections = ["Производство", "Торговля", "Строительство", "Услуги", "ИТ", "Транспорт"]
        okved_col = [sections[int(hashlib.md5(inn.encode()).hexdigest(), 16) % len(sections)]
                     for inn in df["inn"].to_list()]
        df = df.with_columns(pl.Series("okved_section", okved_col))

    if "risk_label" not in df.columns:
        st.info("Запустите скоринг для отображения распределения риска.")
        return

    sector_risk = (
        df.group_by(["okved_section", "risk_label"])
        .agg(pl.len().alias("count"))
        .sort(["okved_section", "risk_label"])
    ).to_pandas()

    fig = px.bar(
        sector_risk,
        x="okved_section",
        y="count",
        color="risk_label",
        color_discrete_map=_RISK_COLORS,
        barmode="stack",
        title="Распределение риск-классов по отраслям",
        labels={"okved_section": "Отрасль", "count": "Число компаний", "risk_label": "Риск"},
    )
    fig.update_layout(height=450, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    pivot = (
        df.group_by(["okved_section", "risk_label"])
        .agg(pl.len().alias("count"))
        .pivot(index="okved_section", on="risk_label", values="count")
        .fill_null(0)
    )
    st.dataframe(pivot.to_pandas(), use_container_width=True, hide_index=True)


def main() -> None:
    """
    Точка входа Streamlit-приложения.
    """
    st.set_page_config(
        page_title="Скоринг финансовой устойчивости",
        layout="wide",
    )
    st.title("Скоринг финансовой устойчивости компаний")

    df = load_data()
    model = load_model_cached()

    if model is None:
        st.info(
            "Обученная модель не найдена. Риск-класс определяется эвристически по Z-score. "
            "Запустите обучение командой из README."
        )

    df = score_dataframe(df, model)

    # Боковая панель - выбор компании
    with st.sidebar:
        st.header("Выбор компании")
        input_mode = st.radio("Способ выбора", ["Ввод ИНН", "Список"])

        inn_list = df["inn"].to_list()

        if input_mode == "Ввод ИНН":
            selected_inn = st.text_input("ИНН", placeholder="Введите ИНН компании")
            if selected_inn and selected_inn not in inn_list:
                st.warning(f"ИНН {selected_inn} не найден в базе.")
                selected_inn = None
        else:
            selected_inn = st.selectbox("Выберите компанию", inn_list)

    # Основная область
    tab_card, tab_all, tab_sectors = st.tabs(["Карточка компании", "Все компании", "Отрасли"])

    with tab_card:
        if selected_inn:
            row_df = df.filter(pl.col("inn") == selected_inn)
            if len(row_df) > 0:
                row = row_df.to_pandas().iloc[0].to_dict()
                render_company_card(row, model, df)
            else:
                st.info("Выберите компанию в боковой панели.")
        else:
            st.info("Выберите компанию в боковой панели.")

    with tab_all:
        render_all_companies_tab(df)

    with tab_sectors:
        render_sectors_tab(df)


if __name__ == "__main__":
    main()
