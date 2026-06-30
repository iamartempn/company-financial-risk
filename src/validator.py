"""
Схемы валидации данных с помощью Pandera.
"""

import logging
import warnings

import pandas as pd
import pandera as pa
import polars as pl
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)


# Схема сырых собранных данных
RawFinancialSchema = DataFrameSchema(
    {
        "inn": Column(
            str,
            nullable=False,
            description="ИНН компании",
        ),
        "revenue": Column(
            float,
            nullable=False,
            checks=Check.greater_than_or_equal_to(0),
            description="Выручка, руб.",
        ),
        "total_assets": Column(
            float,
            nullable=False,
            checks=Check.greater_than(0),
            description="Совокупные активы, руб.",
        ),
        "total_liabilities": Column(
            float,
            nullable=False,
            checks=Check.greater_than_or_equal_to(0),
            description="Суммарные обязательства, руб.",
        ),
        "ebit": Column(
            float,
            nullable=False,
            description="Прибыль до вычета процентов и налогов, руб.",
        ),
        "working_capital": Column(
            float,
            nullable=False,
            description="Оборотный капитал (текущие активы - текущие обязательства), руб.",
        ),
        "retained_earnings": Column(
            float,
            nullable=False,
            description="Нераспределенная прибыль, руб.",
        ),
        "default": Column(
            int,
            nullable=False,
            checks=Check.isin([0, 1]),
            description="Факт дефолта: 0 - нет, 1 - да.",
        ),
    },
    coerce=True,
)


# Схема вычисленных коэффициентов
FeatureSchema = DataFrameSchema(
    {
        "inn": Column(str, nullable=False),
        "z_score": Column(
            float,
            nullable=True,
            checks=Check.in_range(-20, 50),
            description="Модифицированный Z-score Альтмана.",
        ),
        "debt_ratio": Column(
            float,
            nullable=True,
            checks=Check.in_range(0, 100),
            description="Доля заемных средств в активах.",
        ),
        "roa": Column(
            float,
            nullable=True,
            description="Рентабельность активов (EBIT / активы).",
        ),
        "asset_turnover": Column(
            float,
            nullable=True,
            checks=Check.greater_than_or_equal_to(0),
            description="Оборачиваемость активов (выручка / активы).",
        ),
        "working_capital_ratio": Column(
            float,
            nullable=True,
            description="Доля оборотного капитала в активах.",
        ),
        "retained_earnings_ratio": Column(
            float,
            nullable=True,
            description="Доля нераспределенной прибыли в активах.",
        ),
    },
    coerce=True,
)


def validate_raw(df: pl.DataFrame) -> pl.DataFrame:
    """
    Валидирует сырые финансовые данные по схеме RawFinancialSchema.

    Строки, не прошедшие валидацию, удаляются с предупреждением.

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с сырыми данными.

    Возвращает
    ----------
    pl.DataFrame
        Очищенный DataFrame, прошедший валидацию.
    """
    pdf = df.to_pandas()
    pdf = _cast_numeric_columns(pdf, ["revenue", "total_assets", "total_liabilities",
                                      "ebit", "working_capital", "retained_earnings"])
    pdf["default"] = pdf["default"].astype(int)
    pdf["inn"] = pdf["inn"].astype(str)

    try:
        validated = RawFinancialSchema.validate(pdf, lazy=True)
        return pl.from_pandas(validated)
    except pa.errors.SchemaErrors as exc:
        error_rows = set(exc.failure_cases["index"].dropna().astype(int).tolist())
        logger.warning(
            "Удалено %d строк, не прошедших валидацию: %s",
            len(error_rows),
            exc.failure_cases[["column", "check", "failure_case"]].head(10).to_dict("records"),
        )
        clean_pdf = pdf.drop(index=list(error_rows)).reset_index(drop=True)
        return pl.from_pandas(clean_pdf)


def validate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Валидирует DataFrame с вычисленными коэффициентами по схеме FeatureSchema.

    Строки, не прошедшие валидацию, удаляются с предупреждением.

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с финансовыми коэффициентами.

    Возвращает
    ----------
    pl.DataFrame
        Очищенный DataFrame, прошедший валидацию.
    """
    # Оставляем только колонки, описанные в схеме
    schema_cols = list(FeatureSchema.columns.keys())
    present_cols = [c for c in schema_cols if c in df.columns]
    pdf = df.select(present_cols).to_pandas()
    pdf["inn"] = pdf["inn"].astype(str)

    float_cols = [c for c in present_cols if c != "inn"]
    pdf = _cast_numeric_columns(pdf, float_cols)

    try:
        # Используем только присутствующие колонки при валидации
        partial_schema = FeatureSchema.select_columns(present_cols)
        validated = partial_schema.validate(pdf, lazy=True)
        # Возвращаем полный df с исходными колонками, удалив невалидные строки
        valid_idx = validated.index
        return pl.from_pandas(df.to_pandas().loc[valid_idx].reset_index(drop=True))
    except pa.errors.SchemaErrors as exc:
        error_rows = set(exc.failure_cases["index"].dropna().astype(int).tolist())
        logger.warning(
            "Признаки: удалено %d строк с невалидными значениями", len(error_rows)
        )
        full_pdf = df.to_pandas()
        clean_pdf = full_pdf.drop(index=list(error_rows)).reset_index(drop=True)
        return pl.from_pandas(clean_pdf)


def _cast_numeric_columns(pdf: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Приводит указанные колонки к типу float, заменяя непреобразуемые значения на NaN.

    Параметры
    ---------
    pdf : pd.DataFrame
        Исходный DataFrame.
    cols : list
        Список имен колонок.

    Возвращает
    ----------
    pd.DataFrame
        DataFrame с приведенными типами.
    """
    for col in cols:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").astype(float)
    return pdf
