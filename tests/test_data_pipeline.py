"""
Tests de src/data_pipeline.py.

Cubren la lógica que no necesita red: cálculo de retornos, volatilidad rolling
(con la anualización unificada), matriz de correlación, roundtrip parquet y la
paginación de fetch_ohlcv (con un exchange falso, sin tocar la red).

Corre con la stdlib (no requiere pytest):
    python tests/test_data_pipeline.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Los módulos viven en src/ y se importan por nombre (no es un paquete).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_pipeline as dp  # noqa: E402
from metrics import periods_per_year  # noqa: E402


def _ohlcv(closes):
    """DataFrame OHLCV mínimo con un índice de fechas y la columna close dada."""
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        },
        index=idx,
    )


class TestComputeReturns(unittest.TestCase):
    def test_adds_simple_and_log_returns(self):
        df = _ohlcv([100.0, 110.0, 121.0])
        out = dp.compute_returns(df)
        self.assertIn("ret_simple", out.columns)
        self.assertIn("ret_log", out.columns)
        # Primer valor es NaN (no hay previo); luego 10% simple.
        self.assertTrue(np.isnan(out["ret_simple"].iloc[0]))
        self.assertAlmostEqual(out["ret_simple"].iloc[1], 0.10)
        self.assertAlmostEqual(out["ret_log"].iloc[1], np.log(110.0 / 100.0))

    def test_does_not_mutate_input(self):
        df = _ohlcv([100.0, 101.0])
        dp.compute_returns(df)
        self.assertNotIn("ret_simple", df.columns)


class TestRollingVolatility(unittest.TestCase):
    def setUp(self):
        # Retornos variables para que la std rolling no sea cero.
        self.returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01])

    def test_matches_std_times_factor(self):
        expected = self.returns.rolling(3).std() * np.sqrt(
            periods_per_year("crypto", "1d")
        )
        got = dp.rolling_volatility(self.returns, "1d", window=3)
        pd.testing.assert_series_equal(got, expected)

    def test_asset_class_changes_factor(self):
        crypto = dp.rolling_volatility(self.returns, "1d", window=3, asset_class="crypto")
        equity = dp.rolling_volatility(self.returns, "1d", window=3, asset_class="equity")
        ratio = (periods_per_year("crypto", "1d") / periods_per_year("equity", "1d")) ** 0.5
        valid = crypto.dropna().index
        np.testing.assert_allclose(
            crypto.loc[valid].values, equity.loc[valid].values * ratio, rtol=1e-9
        )


class TestCorrelationMatrix(unittest.TestCase):
    def test_shape_and_diagonal(self):
        data = {
            "AAA": _ohlcv([100.0, 102.0, 101.0, 104.0, 103.0]),
            "BBB": _ohlcv([50.0, 49.0, 51.0, 50.0, 52.0]),
        }
        corr = dp.correlation_matrix(data)  # calcula ret_log internamente
        self.assertEqual(list(corr.columns), ["AAA", "BBB"])
        self.assertEqual(corr.shape, (2, 2))
        # La diagonal de una matriz de correlación es 1.
        np.testing.assert_allclose(np.diag(corr.values), [1.0, 1.0], atol=1e-9)


class TestParquetRoundtrip(unittest.TestCase):
    def test_save_and_load(self):
        df = dp.compute_returns(_ohlcv([100.0, 101.0, 102.0]))
        original = dp.DATA_DIR
        with tempfile.TemporaryDirectory() as tmp:
            dp.DATA_DIR = Path(tmp)
            try:
                dp.save_parquet({"BTC/MXN": df}, "bitso", "1d")
                loaded = dp.load_parquet("bitso", "BTC/MXN", "1d")
            finally:
                dp.DATA_DIR = original
        # parquet no preserva el atributo freq del índice; los datos sí.
        pd.testing.assert_frame_equal(loaded, df, check_freq=False)


class _FakeExchange:
    """Exchange ccxt mínimo para probar la paginación sin red."""

    rateLimit = 0

    def __init__(self, candles):
        self._candles = candles  # lista de [ts_ms, o, h, l, c, v]

    def parse_timeframe(self, timeframe):
        return 60  # segundos por vela (1m)

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        rows = [c for c in self._candles if c[0] >= (since or 0)]
        return rows[:limit]


class TestFetchOhlcvPagination(unittest.TestCase):
    def test_paginates_and_dedupes(self):
        candles = [
            [0, 1, 1, 1, 1, 10],
            [60000, 2, 2, 2, 2, 11],
            [120000, 3, 3, 3, 3, 12],
        ]
        ex = _FakeExchange(candles)
        # limit=2 obliga a una segunda página para traer la tercera vela.
        df = dp.fetch_ohlcv(ex, "BTC/MXN", timeframe="1m", since=0, limit=2)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(df.index.name, "timestamp")
        # Índice ordenado y datetime.
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertEqual(df["close"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
