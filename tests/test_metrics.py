"""
Tests de la anualización unificada en metrics.py.

Corre con la stdlib (no requiere pytest):
    python tests/test_metrics.py
O, si tienes pytest:
    pytest tests/test_metrics.py
"""

import os
import sys
import unittest
import warnings

import pandas as pd

# Los módulos viven en src/ y se importan por nombre (no es un paquete).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (  # noqa: E402
    ANNUALIZATION_PRESETS,
    BARS_PER_DAY,
    SESSION_BASED,
    annualized_volatility,
    daily_returns,
    max_drawdown,
    periods_per_year,
    sharpe_ratio,
    summary_table,
)


class TestPeriodsPerYear(unittest.TestCase):
    def test_defaults_to_daily_crypto(self):
        # Firma por defecto: crypto + 1d.
        self.assertEqual(periods_per_year(), 365)

    def test_equity_daily_is_252(self):
        self.assertEqual(periods_per_year("equity", "1d"), 252)

    def test_is_product_of_both_axes(self):
        # El factor es preset[activo] * barras_por_día[timeframe].
        for asset_class in ANNUALIZATION_PRESETS:
            for timeframe in BARS_PER_DAY:
                expected = ANNUALIZATION_PRESETS[asset_class] * BARS_PER_DAY[timeframe]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertAlmostEqual(
                        periods_per_year(asset_class, timeframe), expected
                    )

    def test_reproduces_original_crypto_factors(self):
        # Valores que tenía el dict ANNUALIZATION (crypto 24/7) antes de unificar.
        original = {
            "1m": 365 * 24 * 60,
            "5m": 365 * 24 * 12,
            "15m": 365 * 24 * 4,
            "1h": 365 * 24,
            "4h": 365 * 6,
            "1d": 365,
        }
        for tf, expected in original.items():
            self.assertEqual(periods_per_year("crypto", tf), expected)

    def test_unknown_asset_class_raises(self):
        with self.assertRaises(KeyError):
            periods_per_year("bonds", "1d")

    def test_unknown_timeframe_raises(self):
        with self.assertRaises(KeyError):
            periods_per_year("crypto", "3d")

    def test_intraday_session_based_warns(self):
        # equity/fx intradía sobrestima el factor (base 24h) -> debe avisar.
        for asset_class in SESSION_BASED:
            with self.assertWarns(UserWarning):
                periods_per_year(asset_class, "1h")

    def test_daily_session_based_does_not_warn(self):
        for asset_class in SESSION_BASED:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # cualquier aviso se vuelve excepción
                periods_per_year(asset_class, "1d")

    def test_intraday_crypto_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            periods_per_year("crypto", "1m")


class TestSummaryTable(unittest.TestCase):
    COLUMNS = ["Retorno Total", "Volatilidad Anual", "Sharpe Ratio", "Max Drawdown"]

    def setUp(self):
        # Precios deterministas con una caída intermedia para que el drawdown
        # sea distinto de cero.
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        self.prices = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 99.0, 120.0, 115.0],
                "BBB": [50.0, 52.0, 51.0, 53.0, 55.0],
            },
            index=idx,
        )

    def test_shape_and_columns(self):
        out = summary_table(self.prices)
        self.assertEqual(list(out.columns), self.COLUMNS)
        # Una fila por activo, índice = nombres de los activos.
        self.assertEqual(list(out.index), ["AAA", "BBB"])
        self.assertEqual(out.shape, (2, 4))

    def test_matches_component_functions(self):
        # Cada columna debe coincidir con su función base (redondeada a 4).
        out = summary_table(self.prices)
        ret = daily_returns(self.prices)
        vol = annualized_volatility(ret).round(4)
        shp = sharpe_ratio(ret).round(4)
        mdd = max_drawdown(self.prices).round(4)
        for asset in self.prices.columns:
            self.assertAlmostEqual(out.loc[asset, "Volatilidad Anual"], vol[asset])
            self.assertAlmostEqual(out.loc[asset, "Sharpe Ratio"], shp[asset])
            self.assertAlmostEqual(out.loc[asset, "Max Drawdown"], mdd[asset])

    def test_trading_days_scales_volatility(self):
        # 365 vs 252 escala la vol por sqrt(365/252).
        ret = daily_returns(self.prices)
        v252 = annualized_volatility(ret, 252)
        v365 = annualized_volatility(ret, 365)
        factor = (365 / 252) ** 0.5
        for asset in self.prices.columns:
            self.assertAlmostEqual(v365[asset], v252[asset] * factor, places=10)
        # Y la tabla refleja el parámetro.
        out252 = summary_table(self.prices, trading_days=252)
        out365 = summary_table(self.prices, trading_days=365)
        self.assertGreater(
            out365.loc["AAA", "Volatilidad Anual"],
            out252.loc["AAA", "Volatilidad Anual"],
        )

    def test_higher_risk_free_lowers_sharpe(self):
        low = summary_table(self.prices, risk_free_rate=0.0)
        high = summary_table(self.prices, risk_free_rate=0.10)
        for asset in self.prices.columns:
            self.assertGreater(
                low.loc[asset, "Sharpe Ratio"], high.loc[asset, "Sharpe Ratio"]
            )

    def test_max_drawdown_non_positive(self):
        out = summary_table(self.prices)
        self.assertTrue((out["Max Drawdown"] <= 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
