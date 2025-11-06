# ARIMA + (G)ARCH modelling for gold

Este repositorio proporciona utilidades para descargar la serie histórica del oro y entrenar
modelos ARIMA combinados con procesos de volatilidad GARCH y GJR-GARCH. El código está escrito
en Python y se ejecuta desde la línea de comandos.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

El script principal descarga los precios del oro desde Yahoo Finance (símbolo por defecto
`GC=F`), calcula los rendimientos logarítmicos y ajusta dos modelos:

1. ARIMA + GARCH simétrico.
2. ARIMA + GJR-GARCH (Threshold GARCH).

Los resultados se guardan en la carpeta `outputs/` por defecto.

```bash
python -m src.main --start 2015-01-01 --end 2024-01-01 --horizon 20
```

Argumentos principales:

- `--start` y `--end`: Fechas en formato ISO (opcional).
- `--symbol`: Símbolo de Yahoo Finance (por defecto `GC=F`).
- `--arima`: Orden ARIMA como lista separada por comas, por ejemplo `1,0,1`.
- `--garch`: Orden (p, q) del componente GARCH, por ejemplo `1,1`.
- `--gjr`: Orden (p, o, q) del componente GJR-GARCH.
- `--horizon`: Número de pasos para el pronóstico de volatilidad.
- `--output`: Directorio donde guardar resúmenes y pronósticos.

Cada ejecución genera tres archivos por modelo:

- `*_forecast.csv`: Pronósticos de la media (rendimientos) y volatilidad condicional.
- `*_arima.txt`: Resumen del ajuste ARIMA.
- `*_volatility.txt`: Resumen del modelo de volatilidad (GARCH o GJR-GARCH).

## Referencias

- [Bhavya11 – Gold Series Analysis](https://rpubs.com/Bhavya11/gold-seies-analysis)
- [Forecast Gold Price (GitHub)](https://github.com/Alexjrch/Forecast-Gold-Price-arima-garch-lstm-)
