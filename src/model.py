"""
Обучение и применение LightGBM-модели для скоринга риска дефолта.
"""

import pickle
import logging

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

# Признаки, используемые моделью
FEATURE_COLS = [
    "z_score",
    "debt_ratio",
    "roa",
    "asset_turnover",
    "working_capital_ratio",
    "retained_earnings_ratio",
]

# Пороги для присвоения риск-класса
_THRESHOLD_LOW = 0.35
_THRESHOLD_HIGH = 0.65


def train(df: pl.DataFrame, model_path: str) -> None:
    """
    Обучает LightGBM-классификатор на данных с вычисленными признаками.

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с признаками из FEATURE_COLS и колонкой 'default'.
    model_path : str
        Путь для сохранения обученной модели (.pkl).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Отсутствуют признаки: %s. Обучение без них.", missing)

    pdf = df.select(available + ["default"]).to_pandas().dropna()

    X = pdf[available].values
    y = pdf["default"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Нет дефолта", "Дефолт"])
    print(report)

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": available}, f)

    logger.info("Модель сохранена: %s", model_path)


def load_model(model_path: str):
    """
    Загружает обученную модель из файла.

    Параметры
    ---------
    model_path : str
        Путь к файлу .pkl с сохраненной моделью.

    Возвращает
    ----------
    dict
        Словарь с ключами 'model' (LGBMClassifier) и 'feature_cols' (list).
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Модель загружена из %s", model_path)
    return bundle


def score(df: pl.DataFrame, model) -> pl.DataFrame:
    """
    Рассчитывает вероятность дефолта и присваивает риск-класс.

    Параметры
    ---------
    df : pl.DataFrame
        DataFrame с признаками.
    model : dict
        Словарь с ключами 'model' и 'feature_cols', возвращаемый load_model.

    Возвращает
    ----------
    pl.DataFrame
        Исходный DataFrame с добавленными колонками:
        - risk_score : float, вероятность дефолта [0, 1]
        - risk_label : str, "Низкий" / "Средний" / "Высокий"
    """
    lgbm = model["model"]
    feature_cols = model["feature_cols"]

    pdf = df.to_pandas()
    available = [c for c in feature_cols if c in pdf.columns]

    X = pdf[available].fillna(0).values
    proba = lgbm.predict_proba(X)[:, 1]

    labels = np.where(
        proba < _THRESHOLD_LOW,
        "Низкий",
        np.where(proba < _THRESHOLD_HIGH, "Средний", "Высокий"),
    )

    return df.with_columns([
        pl.Series("risk_score", proba.tolist()),
        pl.Series("risk_label", labels.tolist()),
    ])
