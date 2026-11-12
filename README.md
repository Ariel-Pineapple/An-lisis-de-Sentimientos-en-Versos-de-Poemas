# Análisis de sentimientos en versos de poemas

Proyecto del Diplomado en Inteligencia Artificial del Centro de Investigación en Computación (CIC–IPN). **Autor: Luis Ariel Vázquez Piña.**

El proyecto compara tres modelos supervisados para clasificar versos individuales de poesía en inglés. La aplicación de Streamlit permite comparar sus predicciones sobre un verso o sobre varias líneas de un poema.

## Datos y etiquetas

Se utiliza [Poem Sentiment](https://huggingface.co/datasets/google-research-datasets/poem_sentiment). Se conservan las particiones originales y se filtra la clase `mixed`. Quedan **843 versos de entrenamiento, 105 de validación y 104 de prueba**.

| ID | Etiqueta del dataset | Clase |
|---:|---|---|
| 0 | `negative` | Negativo |
| 1 | `positive` | Positivo |
| 2 | `no_impact` | Sin impacto afectivo |

## Modelos y preprocesamiento

| Modelo | Representación del verso | Tratamiento del desbalance |
|---|---|---|
| TF-IDF + regresión logística | Minúsculas, unigramas y bigramas. El vectorizador forma parte del pipeline. | El notebook incluye variantes con y sin ponderación de clases. |
| BiLSTM | Tokenización, vocabulario de entrenamiento y conversión a índices. Se usan tokens para elementos desconocidos y relleno, con truncamiento a la longitud configurada. | Entropía cruzada ponderada por clase. |
| DistilBERT | Tokenizador de `distilbert/distilbert-base-uncased`, con relleno, truncamiento y máscara de atención. | Entropía cruzada ponderada por clase durante el ajuste. |

El **F1 macro** da el mismo peso al F1 de cada clase. Para medir el efecto de la ponderación se necesitan comparaciones controladas sobre las mismas particiones y métricas por clase.

## Resultados de prueba

Valores recalculados desde los CSV versionados en [`artifacts/`](artifacts/). Los tres modelos se evaluaron sobre los mismos 104 versos, con 19 negativos, 16 positivos y 69 sin impacto afectivo.

| Modelo | F1 macro | Exactitud | Aciertos |
|---|---:|---:|---:|
| TF-IDF + regresión logística | 0.4400 | 63.46% | 66/104 |
| BiLSTM | 0.4574 | 64.42% | 67/104 |
| DistilBERT | **0.8140** | **85.58%** | **89/104** |

La referencia que predice siempre la clase mayoritaria obtiene **66.35% de exactitud y 0.2659 de F1 macro**. DistilBERT obtuvo el mejor F1 macro en estas ejecuciones y reconoció 10 de los 16 versos positivos. La comparación describe las ejecuciones guardadas.

Para auditar una matriz de confusión, usar las columnas `label` y `prediccion` de `predicciones_test.csv` con orden fijo `[0, 1, 2]`, clases reales en filas y predichas en columnas. Contrastar la diagonal, los totales por clase y las métricas derivadas con `resultados.json`.

## Notebooks y ejecución local

Desde la raíz del repositorio, ejecutar los notebooks según el modelo que se quiera reproducir:

1. [Exploración de datos](01_exploracion_datos.ipynb)
2. [TF-IDF + regresión logística](01_exploracion_y_modelo_clasico.ipynb)
3. [BiLSTM](02_modelo_bilstm.ipynb)
4. [DistilBERT](03_modelo_distilbert.ipynb)

Usar las versiones de dependencias compatibles con el entorno que generó los artefactos. Para abrir el prototipo desde ese entorno:

```bash
python -m streamlit run app.py
```

La inferencia requiere generar con los notebooks o recuperar del entorno de entrenamiento los pesos correspondientes:

| Modelo | Artefacto local requerido |
|---|---|
| Clásico | `artifacts/modelo_clasico/pipeline.joblib` |
| BiLSTM | `artifacts/modelo_bilstm/modelo.pt` |
| DistilBERT | Pesos, configuración y tokenizador en `artifacts/modelo_distilbert/` |

Git versiona métricas y predicciones, y excluye los pesos. El `Dockerfile` actual arranca `demo_local.py`.

## Evidencias y trabajo pendiente

- [Historial de BiLSTM](artifacts/modelo_bilstm/historial.csv) e [historial de DistilBERT](artifacts/modelo_distilbert/historial.csv). Usarlos para generar las curvas por época y documentar el criterio de selección.
- Documentar el preprocesamiento con un mismo verso de ejemplo y medir el truncamiento en cada red.
- Comparar el entrenamiento con y sin ponderación usando validación y registrar el efecto por clase.
- Verificar que las matrices del informe y de Streamlit se generen a partir de la misma ejecución que las métricas publicadas.

El alcance evaluado se limita a versos en inglés y a las tres clases indicadas. Al analizar un poema, la interfaz procesa cada línea de forma independiente.
