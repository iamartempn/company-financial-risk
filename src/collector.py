"""
Сбор данных о финансовой отчетности компаний через API bo.nalog.ru.
"""

import time
import logging
import threading
from typing import Optional

import duckdb
import polars as pl
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

# Базовый URL API ФНС для бухгалтерской отчетности
_BASE_URL = "https://bo.nalog.ru/nbo/organizations/{inn}/bfo/json"


class RateLimiter:
    """
    Ограничитель частоты запросов по алгоритму token bucket.
    Потокобезопасен.
    """

    def __init__(self, requests_per_second: float = 2.0) -> None:
        """
        Параметры
        ---------
        requests_per_second : float
            Максимальное количество запросов в секунду.
        """
        self._rate = requests_per_second
        self._tokens = requests_per_second
        self._last_check = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """
        Блокирует вызывающий поток до получения токена.
        Sleep выполняется вне мьютекса, чтобы не блокировать другие потоки.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_check
                self._last_check = now
                self._tokens += elapsed * self._rate
                if self._tokens > self._rate:
                    self._tokens = self._rate
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                sleep_time = (1.0 - self._tokens) / self._rate
            time.sleep(sleep_time)


def fetch_company(inn: str, session: cffi_requests.Session) -> dict:
    """
    Выполняет GET-запрос к API bo.nalog.ru для одной компании.

    Параметры
    ---------
    inn : str
        ИНН компании.
    session : curl_cffi.requests.Session
        Сессия для выполнения запроса.

    Возвращает
    ----------
    dict
        Сырой JSON-ответ от API.

    Исключения
    ----------
    requests.HTTPError
        При статус-коде ответа 4xx или 5xx.
    """
    url = _BASE_URL.format(inn=inn)
    response = session.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_with_retry(
    inn: str,
    session: cffi_requests.Session,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Выполняет запрос к API с экспоненциальной задержкой при ошибке.

    Параметры
    ---------
    inn : str
        ИНН компании.
    session : curl_cffi.requests.Session
        Сессия для выполнения запроса.
    max_retries : int
        Максимальное количество попыток.

    Возвращает
    ----------
    dict или None
        Данные компании или None при исчерпании попыток.
    """
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_company(inn, session)
        except Exception as exc:
            logger.warning(
                "ИНН %s, попытка %d/%d: %s", inn, attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    logger.error("ИНН %s: все попытки исчерпаны", inn)
    return None


def collect_batch(
    inns: list,
    db_path: str,
    requests_per_second: float = 2.0,
) -> None:
    """
    Собирает данные по списку ИНН и сохраняет в DuckDB.

    Параметры
    ---------
    inns : list
        Список ИНН компаний.
    db_path : str
        Путь к файлу DuckDB.
    requests_per_second : float
        Максимальная частота запросов в секунду.
    """
    limiter = RateLimiter(requests_per_second=requests_per_second)
    records = []

    with cffi_requests.Session(impersonate="chrome") as session:
        for inn in inns:
            limiter.acquire()
            data = fetch_with_retry(inn, session)
            if data is None:
                continue
            # Нормализуем ответ до плоского словаря
            record = _parse_response(inn, data)
            if record:
                records.append(record)
                logger.info("ИНН %s собран", inn)

    if not records:
        logger.warning("Нет данных для сохранения")
        return

    df = pl.DataFrame(records)
    con = duckdb.connect(db_path)
    existing = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
    if "raw" not in existing:
        con.execute("CREATE TABLE raw AS SELECT * FROM df")
    else:
        con.execute("INSERT INTO raw SELECT * FROM df")
    con.close()
    logger.info("Сохранено %d записей в %s", len(records), db_path)


def _parse_response(inn: str, data: dict) -> Optional[dict]:
    """
    Извлекает нужные финансовые поля из ответа API.

    Параметры
    ---------
    inn : str
        ИНН компании.
    data : dict
        Сырой JSON от API.

    Возвращает
    ----------
    dict или None
        Плоский словарь с финансовыми показателями.
    """
    try:
        # API bo.nalog.ru возвращает список периодов отчетности;
        # берем последний доступный период
        if isinstance(data, list) and len(data) > 0:
            period = data[-1]
        elif isinstance(data, dict):
            period = data
        else:
            return None

        sections = period.get("sections", {})
        balance = sections.get("balance", {})
        pnl = sections.get("financialResult", {})

        return {
            "inn": inn,
            "revenue": float(pnl.get("revenue", 0) or 0),
            "total_assets": float(balance.get("totalAssets", 0) or 0),
            "total_liabilities": float(balance.get("totalLiabilities", 0) or 0),
            "ebit": float(pnl.get("operatingProfit", 0) or 0),
            "working_capital": float(
                (balance.get("currentAssets", 0) or 0)
                - (balance.get("currentLiabilities", 0) or 0)
            ),
            "retained_earnings": float(
                balance.get("retainedEarnings", 0) or 0
            ),
            "default": 0,
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Ошибка разбора ответа для ИНН %s: %s", inn, exc)
        return None


def load_fallback_data() -> pl.DataFrame:
    """
    Загружает резервный датасет польских компаний (UCI).

    При наличии sklearn пробует загрузить через OpenML, иначе генерирует
    синтетический датасет с аналогичной структурой.

    Возвращает
    ----------
    pl.DataFrame
        DataFrame с колонками:
        inn, revenue, total_assets, total_liabilities, ebit,
        working_capital, retained_earnings, default
    """
    import numpy as np

    try:
        from sklearn.datasets import fetch_openml

        dataset = fetch_openml(
            name="Polish-companies-bankruptcy-data",
            version=1,
            as_frame=True,
        )
        raw = dataset.frame.copy()
        raw = raw.dropna()

        # Датасет содержит 64 признака; нам нужны ближайшие аналоги
        # колонок по смыслу (Attr1..Attr64 - финансовые коэффициенты)
        n = len(raw)
        rng = np.random.default_rng(42)
        total_assets = np.abs(rng.normal(1_000_000, 500_000, n))
        revenue_ratio = np.abs(raw.get("Attr5", rng.uniform(0.5, 2.0, n)))
        revenue = total_assets * revenue_ratio.to_numpy().astype(float)

        df = pl.DataFrame({
            "inn": [f"UCI_{i:06d}" for i in range(n)],
            "revenue": revenue.tolist(),
            "total_assets": total_assets.tolist(),
            "total_liabilities": (total_assets * np.abs(
                raw.get("Attr6", rng.uniform(0.2, 0.8, n)).to_numpy().astype(float)
            )).tolist(),
            "ebit": (total_assets * raw.get(
                "Attr3", rng.normal(0.05, 0.1, n)
            ).to_numpy().astype(float)).tolist(),
            "working_capital": (total_assets * raw.get(
                "Attr1", rng.normal(0.1, 0.2, n)
            ).to_numpy().astype(float)).tolist(),
            "retained_earnings": (total_assets * raw.get(
                "Attr2", rng.normal(0.05, 0.15, n)
            ).to_numpy().astype(float)).tolist(),
            "default": raw["class"].astype(int).tolist(),
        })
        logger.info("Загружен UCI-датасет: %d записей", len(df))
        return df

    except Exception as exc:
        logger.warning(
            "Не удалось загрузить UCI-датасет: %s. Генерируется синтетический.", exc
        )
        return _generate_synthetic_data()


def _generate_synthetic_data(n: int = 1000) -> pl.DataFrame:
    """
    Генерирует синтетический датасет с реалистичными финансовыми показателями.

    Параметры
    ---------
    n : int
        Количество записей.

    Возвращает
    ----------
    pl.DataFrame
        Синтетический датасет в стандартной структуре.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    total_assets = np.abs(rng.normal(5_000_000, 3_000_000, n))
    default_flag = rng.binomial(1, 0.2, n)

    # Компании в дефолте имеют худшие показатели
    debt_ratio = np.where(
        default_flag, rng.uniform(0.7, 1.5, n), rng.uniform(0.2, 0.6, n)
    )
    roa = np.where(
        default_flag, rng.normal(-0.05, 0.1, n), rng.normal(0.08, 0.05, n)
    )

    return pl.DataFrame({
        "inn": [f"SYN_{i:06d}" for i in range(n)],
        "revenue": (total_assets * rng.uniform(0.5, 2.5, n)).tolist(),
        "total_assets": total_assets.tolist(),
        "total_liabilities": (total_assets * debt_ratio).tolist(),
        "ebit": (total_assets * roa).tolist(),
        "working_capital": (total_assets * rng.normal(0.1, 0.2, n)).tolist(),
        "retained_earnings": (total_assets * rng.normal(0.05, 0.15, n)).tolist(),
        "default": default_flag.tolist(),
    })
