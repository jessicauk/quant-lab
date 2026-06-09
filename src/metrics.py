"""
Módulo de métricas cuantitativas.
Calcula retornos, volatilidad, Sharpe ratio y drawdown.
"""

import pandas as pd
import numpy as np


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos diarios porcentuales."""
    return prices.pct_change().dropna()


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos acumulados (compuestos).
    Usa multiplicación, no suma — como vimos en el ejemplo de AAPL.
    """
    return (1 + returns).cumprod() - 1


def annualized_volatility(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Volatilidad anualizada.
    Multiplica la desviación estándar diaria por raíz de 252.
    """
    return returns.std() * np.sqrt(trading_days)


def sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.05,
                 trading_days: int = 252) -> pd.Series:
    """
    Sharpe Ratio — mide el retorno ajustado por riesgo.

    Fórmula: (retorno anualizado - tasa libre de riesgo) / volatilidad anualizada

    Un Sharpe > 1 es bueno, > 2 es muy bueno, > 3 es excepcional.

    Args:
        returns: DataFrame de retornos diarios
        risk_free_rate: Tasa libre de riesgo anual (default 5%)
        trading_days: Días de trading al año
    """
    ann_return = returns.mean() * trading_days
    ann_vol = annualized_volatility(returns, trading_days)
    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Máximo drawdown — la peor caída desde un pico hasta un valle.

    Si invertiste $100 y en el peor momento bajó a $70,
    tu max drawdown fue -30%. Mide el peor escenario histórico.
    """
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown.min()


def drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula la serie completa de drawdown para graficar."""
    cumulative_max = prices.cummax()
    return (prices - cumulative_max) / cumulative_max


def summary_table(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla resumen con todas las métricas por activo.
    Esta es la tabla que un quant mostraría a su equipo.
    """
    ret = daily_returns(prices)
    cum_ret = cumulative_returns(ret)

    summary = pd.DataFrame({
        "Retorno Total": cum_ret.iloc[-1],
        "Volatilidad Anual": annualized_volatility(ret),
        "Sharpe Ratio": sharpe_ratio(ret),
        "Max Drawdown": max_drawdown(prices)
    })

    return summary.round(4)


if __name__ == "__main__":
    from data import download_data

    tickers = ["SPY", "AAPL", "MSFT", "AMZN"]
    prices = download_data(tickers, start="2020-01-01", end="2024-01-01")

    print("=== Tabla resumen ===")
    print(summary_table(prices))