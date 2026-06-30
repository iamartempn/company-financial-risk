"""
SHAP-объяснения для модели скоринга риска дефолта.
"""

import logging
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import shap

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


def compute_shap(df, model) -> Tuple[np.ndarray, shap.TreeExplainer]:
    """
    Вычисляет SHAP-значения для всех записей DataFrame.

    Параметры
    ---------
    df : pl.DataFrame или pd.DataFrame
        DataFrame с признаками модели.
    model : dict
        Словарь с ключами 'model' и 'feature_cols', возвращаемый load_model.

    Возвращает
    ----------
    tuple : (shap_values, explainer)
        shap_values : np.ndarray форма (n_samples, n_features)
            SHAP-значения для класса "дефолт" (индекс 1).
        explainer : shap.TreeExplainer
            Объект объяснителя для дальнейшего использования.
    """
    lgbm = model["model"]
    feature_cols = model["feature_cols"]

    if _POLARS_AVAILABLE and hasattr(df, "to_pandas"):
        pdf = df.to_pandas()
    else:
        pdf = df

    available = [c for c in feature_cols if c in pdf.columns]
    X = pdf[available].fillna(0).values

    explainer = shap.TreeExplainer(lgbm)
    shap_values_all = explainer.shap_values(X)

    # LightGBM возвращает список [shap_class0, shap_class1] при бинарной классификации
    if isinstance(shap_values_all, list) and len(shap_values_all) == 2:
        shap_values = shap_values_all[1]
    else:
        shap_values = np.array(shap_values_all)

    return shap_values, explainer


def waterfall_fig(shap_values: np.ndarray, idx: int, feature_names: list) -> go.Figure:
    """
    Строит Plotly waterfall-диаграмму вкладов признаков для одной записи.

    Параметры
    ---------
    shap_values : np.ndarray
        Массив SHAP-значений форма (n_samples, n_features).
    idx : int
        Индекс записи в массиве.
    feature_names : list
        Список имен признаков.

    Возвращает
    ----------
    go.Figure
        Plotly-фигура с waterfall-диаграммой.
    """
    values = shap_values[idx]
    # Сортируем по абсолютному вкладу
    order = np.argsort(np.abs(values))[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_values = values[order]

    colors = [
        "rgba(220, 53, 69, 0.85)" if v > 0 else "rgba(40, 167, 69, 0.85)"
        for v in sorted_values
    ]

    fig = go.Figure(go.Bar(
        x=sorted_values.tolist(),
        y=sorted_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in sorted_values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Вклад признаков в оценку риска (SHAP waterfall)",
        xaxis_title="SHAP-значение",
        yaxis_title="Признак",
        height=400,
        margin=dict(l=160, r=40, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
        xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="#888"),
    )

    return fig


def summary_fig(shap_values: np.ndarray, feature_names: list) -> go.Figure:
    """
    Строит Plotly bar-диаграмму средних абсолютных SHAP-значений по всем записям.

    Параметры
    ---------
    shap_values : np.ndarray
        Массив SHAP-значений форма (n_samples, n_features).
    feature_names : list
        Список имен признаков.

    Возвращает
    ----------
    go.Figure
        Plotly-фигура со сводным графиком важности признаков.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    sorted_names = [feature_names[i] for i in order]
    sorted_values = mean_abs[order]

    fig = go.Figure(go.Bar(
        x=sorted_values.tolist(),
        y=sorted_names,
        orientation="h",
        marker_color="rgba(13, 110, 253, 0.75)",
        text=[f"{v:.3f}" for v in sorted_values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Средний абсолютный вклад признаков (SHAP summary)",
        xaxis_title="Среднее |SHAP-значение|",
        yaxis_title="Признак",
        height=400,
        margin=dict(l=160, r=40, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
    )

    return fig
