"""
Módulo de métricas cuantitativas.
Calcula retornos, volatilidad, Sharpe ratio y drawdown.

El factor de anualización (`trading_days`) depende del tipo de activo:
usa ANNUALIZATION_PRESETS o pásale un número propio.
"""

import warnings

import pandas as pd
import numpy as np


# =========================================================================
# Anualización — fuente de verdad única para todo el proyecto
# =========================================================================
# Eje 1: periodos por año en base DIARIA, según la clase de activo.
# Pásale uno de estos a las funciones (trading_days=ANNUALIZATION_PRESETS["crypto"])
# o un entero propio.
ANNUALIZATION_PRESETS = {
    "equity": 252,   # acciones / ETFs (días hábiles de mercado)
    "fx": 252,       # forex (similar a acciones)
    "crypto": 365,   # opera 24/7
    "daily": 365,    # cualquier serie diaria de calendario
}

# Eje 2: cuántas barras de cada timeframe caben en un día de calendario (24h).
# Sirve para anualizar datos intradía o semanales.
BARS_PER_DAY = {
    "1m": 24 * 60,
    "5m": 24 * 12,
    "15m": 24 * 4,
    "1h": 24,
    "4h": 6,
    "1d": 1,
    "1w": 1 / 7,
}

DEFAULT_TRADING_DAYS = ANNUALIZATION_PRESETS["equity"]  # acciones por defecto

# Activos con sesión limitada (no 24/7). Para ellos, BARS_PER_DAY (base 24h)
# sobreestima la anualización intradía; lo señalamos con un aviso.
SESSION_BASED = {"equity", "fx"}


def periods_per_year(asset_class: str = "crypto", timeframe: str = "1d") -> float:
    """
    Factor de anualización combinando clase de activo y timeframe.

    Es el producto de los dos ejes:
        periodos/año = (periodos diarios del activo) * (barras por día del timeframe)

    Ejemplos:
        periods_per_year("equity", "1d")  -> 252
        periods_per_year("crypto", "1d")  -> 365
        periods_per_year("crypto", "1h")  -> 8760
        periods_per_year("crypto", "1w")  -> ~52

    Nota: BARS_PER_DAY asume calendario 24h, exacto para activos 24/7 (cripto).
    Para intradía de acciones (sesión ~6.5h) ajusta el preset o pasa el factor a mano.
    """
    if asset_class not in ANNUALIZATION_PRESETS:
        raise KeyError(f"asset_class desconocido: {asset_class!r}. "
                       f"Opciones: {list(ANNUALIZATION_PRESETS)}")
    if timeframe not in BARS_PER_DAY:
        raise KeyError(f"timeframe desconocido: {timeframe!r}. "
                       f"Opciones: {list(BARS_PER_DAY)}")
    if asset_class in SESSION_BASED and BARS_PER_DAY[timeframe] > 1:
        warnings.warn(
            f"Anualización intradía para '{asset_class}' con timeframe '{timeframe}': "
            f"BARS_PER_DAY asume sesión de 24h, pero este activo opera menos horas, "
            f"así que el factor queda sobrestimado. Para datos diarios usa timeframe '1d'.",
            stacklevel=2,
        )
    return ANNUALIZATION_PRESETS[asset_class] * BARS_PER_DAY[timeframe]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos diarios porcentuales."""
    return prices.pct_change().dropna()


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos acumulados (compuestos).
    Usa multiplicación, no suma — como vimos en el ejemplo de AAPL.
    """
    return (1 + returns).cumprod() - 1


def annualized_volatility(returns: pd.DataFrame,
                          trading_days: int = DEFAULT_TRADING_DAYS) -> pd.Series:
    """
    Volatilidad anualizada.
    Multiplica la desviación estándar diaria por raíz de `trading_days`
    (252 acciones, 365 cripto, etc.).
    """
    return returns.std() * np.sqrt(trading_days)


def sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.05,
                 trading_days: int = DEFAULT_TRADING_DAYS) -> pd.Series:
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


def summary_table(prices: pd.DataFrame,
                  trading_days: int = DEFAULT_TRADING_DAYS,
                  risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Genera una tabla resumen con todas las métricas por activo.
    Esta es la tabla que un quant mostraría a su equipo.

    Para otro tipo de activo cambia `trading_days`, p. ej.:
        summary_table(prices, trading_days=ANNUALIZATION_PRESETS["crypto"])
    """
    ret = daily_returns(prices)
    cum_ret = cumulative_returns(ret)

    summary = pd.DataFrame({
        "Retorno Total": cum_ret.iloc[-1],
        "Volatilidad Anual": annualized_volatility(ret, trading_days),
        "Sharpe Ratio": sharpe_ratio(ret, risk_free_rate, trading_days),
        "Max Drawdown": max_drawdown(prices)
    })

    return summary.round(4)


if __name__ == "__main__":
    from data import download_data

    tickers = ["SPY", "AAPL", "MSFT", "AMZN"]
    prices = download_data(tickers, start="2020-01-01", end="2024-01-01")

    print("=== Tabla resumen ===")
    print(summary_table(prices))