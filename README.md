# Laboratorio de minería de textos 1

Juan Camilo Grisales Arias — agosto de 2026

## Contenido

- `text_mining_lab_1.pdf`: informe final del laboratorio.
- `text_mining_lab_1.py`: código fuente completo y reproducible en Python 3.
- `text_mining_lab_1.ipynb`: cuaderno ejecutado, preparado para Google Colab y Jupyter.
- `pyproject.toml`: metadatos y dependencias del proyecto de Python.
- `assets/`: logotipo institucional y los 15 gráficos generados por el análisis.
- `results/`: archivos CSV y JSON que contienen los resultados completos del análisis.

## Ejecución del código

1. [Instala uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Desde este directorio, ejecuta:

   ```sh
   uv run text_mining_lab_1.py
   ```

`uv` instala automáticamente una versión compatible de Python y las dependencias declaradas por el proyecto. El script descarga copias en texto plano de Project Gutenberg cuando no hay una caché local disponible y, después, vuelve a generar los resultados y los gráficos.

## Google Colab

Abre `text_mining_lab_1.ipynb` directamente en Google Colab. El cuaderno contiene el mismo procedimiento que el script de Python.
