"""
src/visualization.py — Gráficas profesionales (Semana 3)
========================================================
Módulo de visualización del quant-portfolio-analyzer. Toma un DataFrame de
precios (columnas = tickers, índice = fechas) y produce gráficas listas para
el reporte: retornos acumulados, drawdown (underwater), volatilidad rolling y
matriz de correlación.

Conecta con tus otros módulos: puedes pasarle los precios que descarga
`data.py`. Las funciones de cálculo reflejan las de `metrics.py` para que todo
sea consistente.

Nota: la anualización es configurable por tipo de activo mediante el parámetro
`trading_days` (default 252 = acciones). Para cripto usa 365; para datos
intradía, el múltiplo correspondiente. Ver ANNUALIZATION_PRESETS.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fuente de verdad única de la anualización (ver metrics.py).
from metrics import ANNUALIZATION_PRESETS, DEFAULT_TRADING_DAYS  # noqa: F401

# output/ está un nivel arriba de src/
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Paleta consistente para que todas las gráficas combinen
PALETTE = ["#1d6fb8", "#d9822b", "#2a9d63", "#b5483f", "#7159a8", "#5f5e5a"]


def _setup_style() -> None:
    """Estilo limpio y profesional para todas las gráficas."""
    plt.rcParams.update({
        "figure.figsize": (11, 6),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Cálculos (consistentes con metrics.py)
# ---------------------------------------------------------------------------
def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Crecimiento acumulado normalizado a 1 al inicio (base = 1.0)."""
    daily = prices.pct_change().fillna(0)
    return (1 + daily).cumprod()


def drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
    """Drawdown en cada punto: caída desde el máximo previo (<= 0)."""
    cum = cumulative_returns(prices)
    running_max = cum.cummax()
    return cum / running_max - 1


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def plot_cumulative_returns(prices: pd.DataFrame, save: bool = True, show: bool = False):
    _setup_style()
    cum = cumulative_returns(prices)
    fig, ax = plt.subplots()
    for i, col in enumerate(cum.columns):
        ax.plot(cum.index, cum[col], label=col, color=_color(i), linewidth=1.6)
    ax.axhline(1.0, color="#999", linewidth=0.8, linestyle="--")
    ax.set_title("Retornos acumulados (base = 1.0)")
    ax.set_ylabel("Crecimiento de $1 invertido")
    ax.set_xlabel("Fecha")
    ax.legend(ncol=min(len(cum.columns), 4), loc="upper left")
    return _finish(fig, "retornos_acumulados.png", save, show)


def plot_drawdown(prices: pd.DataFrame, save: bool = True, show: bool = False):
    """Gráfica 'underwater': qué tan hundido está cada activo respecto a su máximo."""
    _setup_style()
    dd = drawdown_series(prices)
    fig, ax = plt.subplots()
    for i, col in enumerate(dd.columns):
        ax.plot(dd.index, dd[col] * 100, label=col, color=_color(i), linewidth=1.3)
        ax.fill_between(dd.index, dd[col] * 100, 0, color=_color(i), alpha=0.07)
    ax.set_title("Drawdown — caída desde el máximo previo")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Fecha")
    ax.legend(ncol=min(len(dd.columns), 4), loc="lower left")
    return _finish(fig, "drawdown.png", save, show)


def plot_rolling_volatility(prices: pd.DataFrame, window: int = 21,
                            trading_days: int = DEFAULT_TRADING_DAYS,
                            save: bool = True, show: bool = False):
    """Volatilidad rolling anualizada (ventana en días hábiles, default ~1 mes)."""
    _setup_style()
    daily = prices.pct_change()
    vol = daily.rolling(window).std() * np.sqrt(trading_days)
    fig, ax = plt.subplots()
    for i, col in enumerate(vol.columns):
        ax.plot(vol.index, vol[col] * 100, label=col, color=_color(i), linewidth=1.3)
    ax.set_title(f"Volatilidad rolling anualizada (ventana {window} días)")
    ax.set_ylabel("Volatilidad anualizada (%)")
    ax.set_xlabel("Fecha")
    ax.legend(ncol=min(len(vol.columns), 4), loc="upper left")
    return _finish(fig, "volatilidad_rolling.png", save, show)


def plot_correlation_heatmap(prices: pd.DataFrame, save: bool = True, show: bool = False):
    """Matriz de correlación de retornos diarios, como mapa de calor."""
    _setup_style()
    corr = prices.pct_change().corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    # Anotar cada celda con el valor
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.6 else "#333", fontsize=9)
    ax.set_title("Correlación de retornos diarios")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _finish(fig, "correlacion.png", save, show)


def _finish(fig, fname, save, show):
    fig.tight_layout()
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / fname
        fig.savefig(path, bbox_inches="tight")
        print(f"  Guardada: {path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def generate_all(prices: pd.DataFrame,
                 trading_days: int = DEFAULT_TRADING_DAYS) -> None:
    """
    Genera las cuatro gráficas y las guarda en output/.

    Para otro activo pasa el factor, p. ej.:
        generate_all(prices, trading_days=ANNUALIZATION_PRESETS["crypto"])
    """
    print("Generando gráficas...")
    plot_cumulative_returns(prices)
    plot_drawdown(prices)
    plot_rolling_volatility(prices, trading_days=trading_days)
    plot_correlation_heatmap(prices)
    print("Listo. Revisa la carpeta output/.")


if __name__ == "__main__":
    # Demo: descarga datos y genera todo.
    # Puedes reemplazar esto por tu loader de data.py.
    import yfinance as yf

    tickers = ["SPY", "AAPL", "MSFT", "AMZN"]
    print(f"Descargando {tickers}...")
    prices = yf.download(tickers, start="2020-01-01", end="2024-12-31",
                         auto_adjust=True)["Close"]
    generate_all(prices)