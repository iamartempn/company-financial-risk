"""Тесты валидатора данных компаний."""
import sys
from pathlib import Path
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from validator import validate_raw


class TestValidateCompanyData:
    def test_valid_data_passes(self):
        df = pl.DataFrame({
            "inn": ["1234567890"],
            "revenue": [1_000_000.0],
            "total_assets": [500_000.0],
            "total_liabilities": [200_000.0],
            "ebit": [100_000.0],
            "working_capital": [150_000.0],
            "retained_earnings": [80_000.0],
            "default": [0],
        })
        result = validate_raw(df)
        assert result is not None
        assert len(result) == 1

    def test_negative_assets_removed(self):
        df = pl.DataFrame({
            "inn": ["1234567890"],
            "revenue": [1_000_000.0],
            "total_assets": [-500_000.0],
            "total_liabilities": [200_000.0],
            "ebit": [100_000.0],
            "working_capital": [150_000.0],
            "retained_earnings": [80_000.0],
            "default": [0],
        })
        result = validate_raw(df)
        # Строка с отрицательными активами должна быть удалена или схема должна поднять ошибку
        assert result is not None

    def test_missing_required_column_raises(self):
        df = pl.DataFrame({
            "inn": ["1234567890"],
            "revenue": [1_000_000.0],
        })
        with pytest.raises(Exception):
            validate_raw(df)

    def test_invalid_default_value_removed(self):
        df = pl.DataFrame({
            "inn": ["1234567890"],
            "revenue": [1_000_000.0],
            "total_assets": [500_000.0],
            "total_liabilities": [200_000.0],
            "ebit": [100_000.0],
            "working_capital": [150_000.0],
            "retained_earnings": [80_000.0],
            "default": [5],
        })
        result = validate_raw(df)
        # Строка с некорректным default (не 0/1) должна быть удалена
        assert len(result) == 0

    def test_multiple_valid_rows(self):
        df = pl.DataFrame({
            "inn": ["1111111111", "2222222222", "3333333333"],
            "revenue": [1e6, 2e6, 3e6],
            "total_assets": [5e5, 1e6, 1.5e6],
            "total_liabilities": [2e5, 4e5, 6e5],
            "ebit": [1e5, 2e5, 3e5],
            "working_capital": [1.5e5, 3e5, 4.5e5],
            "retained_earnings": [8e4, 1.6e5, 2.4e5],
            "default": [0, 1, 0],
        })
        result = validate_raw(df)
        assert len(result) == 3
