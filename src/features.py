"""
Вычисление финансовых коэффициентов и нормализация признаков.
"""

import logging
import warnings

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)

# Набор колонок, необходимых для расчета Z-score
_ALTMAN_REQUIRED = [
    "working_capital",
    "retained_earnings",
    "ebit",
    "total_assets",
    "total_liabilities",
    "revenue",
]


def altman_z_score(df: pl.DataFrame) -> pl.Series:
    """
    Рассчитывает модифицированный Z-score Альтмана для непубличных компаний.

    Формула: Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5

    X1 = оборотный капитал / совокупные активы
    X2 = нераспределенная прибыль / совокупные активы
    X3 = EBIT / совокупные активы
    X4 = собственный капитал / суммарные обязательства
    X5 = выручка / совокупные активы

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с финансовыми данными.

    Возвращает
    ----------
    pl.Series
        Значения Z-score. NaN при нулевых активах или отсутствии колонок.
    """
    missing = [col for col in _ALTMAN_REQUIRED if col not in df.columns]
    if missing:
        warnings.warn(
            f"Отсутствуют колонки для Z-score: {missing}. Возвращается NaN.",
            stacklevel=2,
        )
        return pl.Series("z_score", [float("nan")] * len(df))

    assets = df["total_assets"].to_numpy().astype(float)
    liabilities = df["total_liabilities"].to_numpy().astype(float)

    # Защита от деления на ноль
    safe_assets = np.where(np.abs(assets) < 1e-9, np.nan, assets)
    safe_liabilities = np.where(np.abs(liabilities) < 1e-9, np.nan, liabilities)

    x1 = df["working_capital"].to_numpy().astype(float) / safe_assets
    x2 = df["retained_earnings"].to_numpy().astype(float) / safe_assets
    x3 = df["ebit"].to_numpy().astype(float) / safe_assets
    equity = assets - liabilities
    x4 = equity / safe_liabilities
    x5 = df["revenue"].to_numpy().astype(float) / safe_assets

    z = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5

    return pl.Series("z_score", z.tolist())


def compute_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """
    Добавляет колонки с финансовыми коэффициентами.

    Рассчитываемые коэффициенты:
    - z_score : Z-score Альтмана (модифицированный)
    - debt_ratio : суммарные обязательства / совокупные активы
    - roa : EBIT / совокупные активы (рентабельность активов)
    - asset_turnover : выручка / совокупные активы (оборачиваемость)
    - working_capital_ratio : оборотный капитал / совокупные активы
    - retained_earnings_ratio : нераспределенная прибыль / совокупные активы

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с базовыми финансовыми данными.

    Возвращает
    ----------
    pl.DataFrame
        Исходный DataFrame с добавленными колонками коэффициентов.
    """
    assets = df["total_assets"].to_numpy().astype(float)
    safe_assets = np.where(np.abs(assets) < 1e-9, np.nan, assets)

    liabilities = df["total_liabilities"].to_numpy().astype(float)
    ebit = df["ebit"].to_numpy().astype(float)
    revenue = df["revenue"].to_numpy().astype(float)
    wc = df["working_capital"].to_numpy().astype(float)
    re = df["retained_earnings"].to_numpy().astype(float)

    z_score = altman_z_score(df)

    result = df.with_columns([
        z_score.alias("z_score"),
        pl.Series("debt_ratio", (liabilities / safe_assets).tolist()),
        pl.Series("roa", (ebit / safe_assets).tolist()),
        pl.Series("asset_turnover", (revenue / safe_assets).tolist()),
        pl.Series("working_capital_ratio", (wc / safe_assets).tolist()),
        pl.Series("retained_earnings_ratio", (re / safe_assets).tolist()),
    ])

    return result


def normalize_features(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    """
    Применяет z-score нормализацию к указанным колонкам.

    Для каждой колонки: (x - mean) / std. Колонки с нулевым стандартным
    отклонением остаются без изменений.

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с признаками.
    feature_cols : list
        Список имен колонок для нормализации.

    Возвращает
    ----------
    pl.DataFrame
        DataFrame с нормализованными колонками.
    """
    exprs = []
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Колонка '%s' не найдена при нормализации", col)
            continue
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val is None or std_val < 1e-9:
            logger.warning(
                "Колонка '%s': стандартное отклонение равно нулю, пропускается", col
            )
            continue
        exprs.append(
            ((pl.col(col) - mean_val) / std_val).alias(col)
        )

    if not exprs:
        return df

    return df.with_columns(exprs)
