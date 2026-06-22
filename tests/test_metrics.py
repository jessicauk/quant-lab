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

# Los módulos viven en src/ y se importan por nombre (no es un paquete).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (  # noqa: E402
    ANNUALIZATION_PRESETS,
    BARS_PER_DAY,
    SESSION_BASED,
    periods_per_year,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
