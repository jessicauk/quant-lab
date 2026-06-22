# quant-lab

Laboratorio personal de trading cuantitativo. Construido por fases, desde la
ingesta de datos de mercado hasta el backtesting y, eventualmente, paper trading.

Cubre dos universos de activos:

- **Acciones / ETFs** vía [yfinance](https://github.com/ranaroussi/yfinance).
- **Cripto** vía [ccxt](https://github.com/ccxt/ccxt) (bitso, binance, kraken…).

## Fase actual: 1 — Pipeline de datos

Descarga datos OHLCV de mercados cripto y de renta variable, los almacena de
forma reproducible (parquet / CSV) y calcula métricas base: retornos simples y
logarítmicos, retornos acumulados, volatilidad anualizada, Sharpe ratio,
máximo drawdown y correlaciones entre activos.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### Cripto (ccxt)

```bash
# Pares en MXN desde Bitso (historia limitada)
python src/data_pipeline.py --exchange bitso --symbols BTC/MXN ETH/MXN --timeframe 1d --days 365

# Más profundidad histórica desde un exchange grande
python src/data_pipeline.py --exchange binance --symbols BTC/USDT ETH/USDT --timeframe 1d --days 1000
```

El mismo pipeline sirve para otras clases de activo cambiando `--asset-class`,
que ajusta la anualización (252 para acciones/forex, 365 para cripto):

```bash
# Anualizar como renta variable (factor 252)
python src/data_pipeline.py --exchange binance --symbols BTC/USDT --asset-class equity --timeframe 1d
```

#### Parámetros

| Flag | Descripción | Default |
|------|-------------|---------|
| `--exchange` | ID de exchange en ccxt (bitso, binance, kraken…) | `bitso` |
| `--symbols` | Pares a descargar, separados por espacio | `BTC/MXN ETH/MXN` |
| `--timeframe` | Resolución de las velas (1m, 1h, 1d, 1w…) | `1d` |
| `--asset-class` | Clase de activo para anualizar: `equity`, `fx`, `crypto`, `daily` | `crypto` |
| `--days` | Días de historia a descargar | `365` |

El factor de anualización se calcula combinando `--asset-class` (base por año:
`equity`/`fx`=252, `crypto`/`daily`=365) con `--timeframe` (barras por día). Si
anualizas datos **intradía** de un activo con sesión limitada (`equity`/`fx`), el
pipeline avisa: la base de 24 h sobreestima el factor, así que para esos activos
usa `--timeframe 1d`.

Los datos crudos quedan en `data/<exchange>/<timeframe>/` (ignorados por git).

### Acciones / ETFs (yfinance)

```python
from src.data import download_data
from src.metrics import summary_table

prices = download_data(["SPY", "AAPL", "MSFT", "AMZN"],
                       start="2020-01-01", end="2024-01-01")

# Tabla resumen: retorno total, volatilidad anual, Sharpe, max drawdown
print(summary_table(prices))
```

O ejecutando los módulos directamente:

```bash
python src/data.py        # descarga precios y guarda output/prices.csv
python src/metrics.py     # imprime la tabla resumen de métricas
```

## Estructura

```
quant-lab/
├── src/
│   ├── data.py             # Descarga de acciones/ETFs (yfinance)
│   ├── data_pipeline.py    # Pipeline cripto (ccxt): ingesta + métricas base
│   ├── metrics.py          # Métricas de riesgo/retorno + anualización (fuente única)
│   └── visualization.py    # Gráficas (retornos, drawdown, volatilidad, correlación)
├── notebooks/
│   ├── exploration.ipynb                    # Análisis exploratorio
│   └── exploration_emerging_markets.ipynb   # Desarrollados vs. emergentes
├── data/                   # parquet descargado (ignorado por git)
├── output/                 # gráficas y CSVs generados (ignorado por git)
├── tests/
├── requirements.txt
├── requirements.lock       # versiones exactas para reproducibilidad
├── .gitignore
└── README.md
```

## Roadmap

- [x] **Fase 1 — Pipeline de datos:** ingesta, almacenamiento, métricas base
- [ ] **Fase 2 — Backtesting honesto:** costos, slippage, sin lookahead bias
- [ ] **Fase 3 — Modelos de factores y optimización de portafolio**
- [ ] **Fase 4 — Machine learning con validación walk-forward**
- [ ] **Fase 5 — Infraestructura y paper trading**

## Notas

- La anualización es configurable por clase de activo (`--asset-class` en el CLI,
  o el parámetro `trading_days` en `metrics.py`). La fuente de verdad única vive
  en `metrics.py` (`ANNUALIZATION_PRESETS` + `BARS_PER_DAY` + `periods_per_year`):
  **252** para acciones/forex, **365** para cripto (opera 24/7).
- Nunca subas claves de API al repo: van en `.env`, que está en `.gitignore`.
