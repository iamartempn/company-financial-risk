"""Тесты вычисления признаков."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import compute_ratios, altman_z_score


def make_company_df(**kwargs) -> pl.DataFrame:
    defaults = {
        "inn": ["1234567890"],
        "revenue": [1_000_000.0],
        "total_assets": [500_000.0],
        "total_liabilities": [200_000.0],
        "ebit": [100_000.0],
        "working_capital": [150_000.0],
        "retained_earnings": [80_000.0],
        "default": [0],
    }
    defaults.update(kwargs)
    return pl.DataFrame(defaults)


class TestComputeRatios:
    def test_adds_required_columns(self):
        df = make_company_df()
        result = compute_ratios(df)
        for col in ["z_score", "debt_ratio", "roa", "asset_turnover",
                    "working_capital_ratio", "retained_earnings_ratio"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_debt_ratio_formula(self):
        df = make_company_df(total_liabilities=[100.0], total_assets=[400.0])
        result = compute_ratios(df)
        assert abs(result["debt_ratio"][0] - 0.25) < 1e-6

    def test_zero_assets_no_crash(self):
        df = make_company_df(total_assets=[0.0])
        result = compute_ratios(df)
        assert result is not None
        assert len(result) == 1

    def test_negative_ebit_roa(self):
        df = make_company_df(ebit=[-50_000.0], total_assets=[500_000.0])
        result = compute_ratios(df)
        assert result["roa"][0] < 0

    def test_multiple_companies(self):
        df = pl.DataFrame({
            "inn": ["111", "222", "333"],
            "revenue": [1e6, 2e6, 3e6],
            "total_assets": [5e5, 1e6, 1.5e6],
            "total_liabilities": [2e5, 4e5, 6e5],
            "ebit": [1e5, 2e5, 3e5],
            "working_capital": [1.5e5, 3e5, 4.5e5],
            "retained_earnings": [8e4, 1.6e5, 2.4e5],
            "default": [0, 1, 0],
        })
        result = compute_ratios(df)
        assert len(result) == 3


class TestAltmanZScore:
    def test_healthy_company_high_z(self):
        df = make_company_df(
            revenue=[5_000_000.0], total_assets=[1_000_000.0],
            total_liabilities=[100_000.0], ebit=[500_000.0],
            working_capital=[400_000.0], retained_earnings=[300_000.0],
        )
        z = altman_z_score(df)
        assert z[0] > 2.9, f"Expected high Z-score for healthy company, got {z[0]}"

    def test_distressed_company_low_z(self):
        df = make_company_df(
            revenue=[100_000.0], total_assets=[1_000_000.0],
            total_liabilities=[950_000.0], ebit=[-50_000.0],
            working_capital=[-100_000.0], retained_earnings=[-200_000.0],
        )
        z = altman_z_score(df)
        assert z[0] < 1.81, f"Expected low Z-score for distressed company, got {z[0]}"
