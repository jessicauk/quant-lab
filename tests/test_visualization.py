"""
Tests de src/visualization.py.

Usa el backend Agg (sin ventana) y save=False para no escribir archivos.
Corre con la stdlib (no requiere pytest):
    python tests/test_visualization.py
"""

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")  # backend no interactivo, antes de importar pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Los módulos viven en src/ y se importan por nombre (no es un paquete).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import visualization as viz  # noqa: E402


def _make_prices(n=60):
    """Precios deterministas con volatilidad variable (sin aleatoriedad)."""
    t = np.arange(n)
    a = 100 * np.cumprod(1 + 0.01 * np.sin(t / 3))
    b = 100 * np.cumprod(1 + 0.005 * np.cos(t / 4))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


class TestCalculations(unittest.TestCase):
    def test_cumulative_returns_base_one(self):
        prices = pd.DataFrame({"X": [100.0, 110.0, 121.0]})
        cum = viz.cumulative_returns(prices)
        # Normalizado a 1 al inicio, crecimiento compuesto.
        self.assertAlmostEqual(cum["X"].iloc[0], 1.0)
        self.assertAlmostEqual(cum["X"].iloc[1], 1.1)
        self.assertAlmostEqual(cum["X"].iloc[2], 1.21)

    def test_drawdown_non_positive_and_zero_at_peaks(self):
        prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]})
        dd = viz.drawdown_series(prices)
        self.assertTrue((dd["X"] <= 1e-12).all())
        # En máximos sucesivos el drawdown es 0; tras la caída es negativo.
        self.assertAlmostEqual(dd["X"].iloc[0], 0.0)
        self.assertAlmostEqual(dd["X"].iloc[1], 0.0)
        self.assertLess(dd["X"].iloc[2], 0.0)

    def test_color_cycles_through_palette(self):
        n = len(viz.PALETTE)
        self.assertEqual(viz._color(0), viz.PALETTE[0])
        self.assertEqual(viz._color(n), viz.PALETTE[0])  # da la vuelta
        self.assertEqual(viz._color(n + 1), viz.PALETTE[1])


class TestPlots(unittest.TestCase):
    def setUp(self):
        self.prices = _make_prices()

    def tearDown(self):
        plt.close("all")

    def test_plot_functions_return_figure(self):
        for fn in (
            viz.plot_cumulative_returns,
            viz.plot_drawdown,
            viz.plot_rolling_volatility,
            viz.plot_correlation_heatmap,
        ):
            fig = fn(self.prices, save=False, show=False)
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(fig.axes, f"{fn.__name__} no produjo ejes")

    def test_rolling_volatility_uses_trading_days(self):
        # Cambiar trading_days escala la línea de volatilidad por sqrt(365/252).
        f252 = viz.plot_rolling_volatility(
            self.prices, trading_days=252, save=False, show=False
        )
        f365 = viz.plot_rolling_volatility(
            self.prices, trading_days=365, save=False, show=False
        )
        y252 = f252.axes[0].lines[0].get_ydata()
        y365 = f365.axes[0].lines[0].get_ydata()
        mask = ~np.isnan(y252) & ~np.isnan(y365)
        self.assertTrue(mask.any())
        factor = (365 / 252) ** 0.5
        np.testing.assert_allclose(y365[mask], y252[mask] * factor, rtol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
