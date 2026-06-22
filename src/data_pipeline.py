"""
Pipeline de datos de mercado cripto — Fase 1
=============================================
Descarga OHLCV histórico vía ccxt, lo almacena de forma reproducible (parquet)
y calcula métricas base: retornos, volatilidad rolling (anualizada) y correlaciones.

Uso rápido:
    python crypto_pipeline.py --exchange bitso --symbols BTC/MXN ETH/MXN --timeframe 1d --days 365

Requiere: ccxt, pandas, pyarrow  (pip install ccxt pandas pyarrow)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# Anualización: fuente de verdad única en metrics.py (presets por activo +
# barras por timeframe). Este pipeline es de cripto, así que usa el preset 24/7.
from metrics import ANNUALIZATION_PRESETS, BARS_PER_DAY, periods_per_year

# Carpeta donde se guardan los datos crudos (parquet). Versionable o ignorable en git.
DATA_DIR = Path("data")

# Clase de activo de este pipeline (cripto opera 24/7 -> base 365).
ASSET_CLASS = "crypto"


# ---------------------------------------------------------------------------
# 1. Descarga de datos
# ---------------------------------------------------------------------------
def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1d",
    since: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Descarga OHLCV paginando hacia adelante hasta llegar al presente.

    ccxt limita cada llamada (típico 500-1000 velas), así que iteramos usando
    el timestamp de la última vela como nuevo 'since' hasta que ya no haya más.
    """
    all_rows: list[list] = []
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows += batch
        # Avanzamos al timestamp siguiente a la última vela recibida
        since = batch[-1][0] + timeframe_ms
        # Si el exchange devolvió menos del límite, ya no hay más historia
        if len(batch) < limit:
            break
        # Respetamos el rate limit del exchange
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df


def fetch_multiple(
    exchange_id: str,
    symbols: list[str],
    timeframe: str,
    since: int | None,
) -> dict[str, pd.DataFrame]:
    """Descarga varios símbolos del mismo exchange y los devuelve en un dict."""
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in exchange.markets:
            print(f"  [aviso] {symbol} no está disponible en {exchange_id}, lo salto")
            continue
        print(f"  Descargando {symbol} ({timeframe})...")
        df = fetch_ohlcv(exchange, symbol, timeframe, since)
        print(f"    -> {len(df)} velas, de {df.index.min()} a {df.index.max()}")
        data[symbol] = df
    return data


# ---------------------------------------------------------------------------
# 2. Almacenamiento reproducible
# ---------------------------------------------------------------------------
def save_parquet(data: dict[str, pd.DataFrame], exchange_id: str, timeframe: str) -> None:
    """Guarda cada símbolo como parquet. Nombre estable -> reproducible."""
    out = DATA_DIR / exchange_id / timeframe
    out.mkdir(parents=True, exist_ok=True)
    for symbol, df in data.items():
        fname = symbol.replace("/", "-") + ".parquet"
        df.to_parquet(out / fname)
    print(f"  Guardado en {out}")


def load_parquet(exchange_id: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Carga un símbolo previamente guardado."""
    fname = symbol.replace("/", "-") + ".parquet"
    return pd.read_parquet(DATA_DIR / exchange_id / timeframe / fname)


# ---------------------------------------------------------------------------
# 3. Métricas base
# ---------------------------------------------------------------------------
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega retornos simples y logarítmicos sobre el precio de cierre."""
    out = df.copy()
    out["ret_simple"] = out["close"].pct_change()
    out["ret_log"] = np.log(out["close"] / out["close"].shift(1))
    return out


def rolling_volatility(returns: pd.Series, timeframe: str, window: int = 30,
                       asset_class: str = ASSET_CLASS) -> pd.Series:
    """Volatilidad rolling anualizada (desviación estándar * sqrt(factor))."""
    factor = periods_per_year(asset_class, timeframe)
    return returns.rolling(window).std() * np.sqrt(factor)


def correlation_matrix(data: dict[str, pd.DataFrame], use: str = "ret_log") -> pd.DataFrame:
    """Matriz de correlación de retornos entre todos los símbolos (alineados por fecha)."""
    cols = {}
    for symbol, df in data.items():
        if use not in df.columns:
            df = compute_returns(df)
        cols[symbol] = df[use]
    aligned = pd.DataFrame(cols).dropna()
    return aligned.corr()


# ---------------------------------------------------------------------------
# 4. Orquestación
# ---------------------------------------------------------------------------
def run(exchange_id: str, symbols: list[str], timeframe: str, days: int,
        asset_class: str = ASSET_CLASS) -> None:
    print(f"\n[1/4] Descargando datos de {exchange_id}...")
    since = ccxt.Exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    data = fetch_multiple(exchange_id, symbols, timeframe, since)
    if not data:
        print("No se descargó ningún símbolo. Revisa nombres de pares / exchange.")
        return

    print("\n[2/4] Guardando en parquet...")
    save_parquet(data, exchange_id, timeframe)

    print(f"\n[3/4] Calculando retornos y volatilidad (activo: {asset_class})...")
    data = {s: compute_returns(df) for s, df in data.items()}
    for symbol, df in data.items():
        vol = rolling_volatility(df["ret_log"], timeframe, asset_class=asset_class).iloc[-1]
        ann_ret = df["ret_log"].mean() * periods_per_year(asset_class, timeframe)
        print(f"  {symbol}: vol anualizada (30) = {vol:.2%} | retorno anualizado ~ {ann_ret:.2%}")

    print("\n[4/4] Matriz de correlación (retornos log):")
    if len(data) > 1:
        print(correlation_matrix(data).round(3).to_string())
    else:
        print("  (necesitas 2+ símbolos para correlación)")
    print("\nListo. Datos crudos disponibles en ./data para la Fase 2 (backtesting).")


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline de datos cripto — Fase 1")
    p.add_argument("--exchange", default="bitso", help="ID de exchange en ccxt (bitso, binance, kraken...)")
    p.add_argument("--symbols", nargs="+", default=["BTC/MXN", "ETH/MXN"], help="Pares a descargar")
    p.add_argument("--timeframe", default="1d", choices=list(BARS_PER_DAY.keys()))
    p.add_argument("--asset-class", default=ASSET_CLASS, choices=list(ANNUALIZATION_PRESETS.keys()),
                   help="Clase de activo para la anualización (equity=252, crypto=365...)")
    p.add_argument("--days", type=int, default=365, help="Cuántos días de historia descargar")
    args = p.parse_args()
    run(args.exchange, args.symbols, args.timeframe, args.days, args.asset_class)


if __name__ == "__main__":
    main()