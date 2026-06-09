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

#### Parámetros

| Flag | Descripción | Default |
|------|-------------|---------|
| `--exchange` | ID de exchange en ccxt (bitso, binance, kraken…) | `bitso` |
| `--symbols` | Pares a descargar, separados por espacio | `BTC/MXN ETH/MXN` |
| `--timeframe` | Resolución de las velas (1m, 1h, 1d, 1w…) | `1d` |
| `--days` | Días de historia a descargar | `365` |

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
│   └── metrics.py          # Métricas de riesgo/retorno (Sharpe, drawdown…)
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

- Para acciones se anualiza con **252** días de trading; cripto opera 24/7, por
  eso usa factor **365**.
- Nunca subas claves de API al repo: van en `.env`, que está en `.gitignore`.
