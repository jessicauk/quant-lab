"""
Módulo de descarga y limpieza de datos financieros.
Usa yfinance para obtener datos OHLCV históricos.
"""

import yfinance as yf
import pandas as pd


def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados para una lista de tickers.

    Args:
        tickers: Lista de símbolos (ej: ["SPY", "AAPL", "MSFT"])
        start: Fecha inicio en formato "YYYY-MM-DD"
        end: Fecha fin en formato "YYYY-MM-DD"

    Returns:
        DataFrame con precios de cierre ajustados, una columna por ticker.
    """
    data = yf.download(tickers, start=start, end=end)

    # Si descargamos varios tickers, yfinance retorna MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    # Eliminar filas con datos faltantes
    prices = prices.dropna()

    return prices


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Guarda el DataFrame como CSV."""
    df.to_csv(filepath)
    print(f"Datos guardados en: {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """Carga datos desde un CSV."""
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


# Permite probar el módulo directamente
if __name__ == "__main__":
    tickers = ["SPY", "AAPL", "MSFT", "AMZN"]
    prices = download_data(tickers, start="2024-01-01", end="2026-04-01")

    print("=== Primeras filas ===")
    print(prices.head())

    print(f"\nPeriodo: {prices.index[0].date()} a {prices.index[-1].date()}")
    print(f"Días de trading: {len(prices)}")
    print(f"Tickers: {prices.columns.tolist()}")

    save_data(prices, "output/prices.csv")