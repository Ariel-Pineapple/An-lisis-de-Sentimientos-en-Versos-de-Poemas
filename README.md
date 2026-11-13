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


## Monitoreo del clasificador de sentimientos en versos

En la Actividad 8 usamos el clasificador de versos como caso de estudio para
explicar cómo observar un modelo en producción, detectar incidentes y comprobar
si una intervención funcionó.

| Concepto | Qué significa | Ejemplo aplicado al proyecto |
|---|---|---|
| **Observabilidad** | Capacidad de entender qué ocurre en el sistema a partir de métricas, logs y trazas. | Una métrica muestra que la predicción tarda más. Una traza permite ubicar la demora en la tokenización o en la inferencia. |
| **Monitorización proactiva** | Seguimiento continuo de señales para detectar problemas antes de que su impacto aumente. | Detectar que la latencia sube durante varias ventanas antes de recibir reportes de usuarios. |
| **SLI, SLO y SLA** | El **SLI** es el indicador medido. El **SLO** es su objetivo. El **SLA** es el compromiso formal con el cliente y puede establecer consecuencias si se incumple. | **SLI:** porcentaje de solicitudes válidas que terminan con éxito en 500 ms o menos. **SLO de ejemplo:** alcanzar el 99 % en 30 días. **SLA:** aplicaría si ese nivel se formalizara con un cliente. |
| **Alertamiento inteligente** | Reglas que consideran cuánto dura una anomalía, cuántas solicitudes afecta y qué impacto tiene. | Disparar una alerta si aumenta la proporción de respuestas tardías durante dos ventanas consecutivas con tráfico suficiente. |
| **Golden Signals** | Cuatro señales de salud del servicio: latencia, tráfico, errores y saturación. | Medir latencia p95, solicitudes por minuto, porcentaje de fallas y uso de memoria o tamaño de la cola. |
| **Dashboard operativo** | Tablero que reúne las señales necesarias para decidir si hace falta intervenir. | Mostrar la latencia, el volumen de solicitudes, la versión activa del modelo y las alertas abiertas. |
| **SRE (*Site Reliability Engineering*)** | Disciplina que aplica ingeniería de software a la confiabilidad y operación de servicios. | Definir quién atiende una alerta, cómo revierte un cambio y qué medición confirma la recuperación. |
| **Monitorización de sistemas distribuidos** | Seguimiento de una solicitud a través de varios componentes mediante métricas, logs y trazas correlacionadas. | En un despliegue con componentes separados, seguir una predicción desde la interfaz hasta la API y el servicio de inferencia con un identificador de traza. |
| **Data drift y model drift** | El *data drift* es un cambio en la distribución de las entradas. El término *model drift* suele usarse para describir el deterioro del desempeño del modelo con el tiempo. | Recibir textos mucho más largos que los de referencia indica un posible *data drift*. Una caída del F1 macro en ejemplos recientes etiquetados aporta evidencia de degradación. |
| **MLOps y monitoreo en producción** | MLOps abarca el versionado, la validación, el despliegue y la operación de modelos. El monitoreo comprueba cómo se comporta el modelo una vez desplegado. | Identificar qué versión de DistilBERT está activa, registrar sus métricas y validar un modelo candidato antes de promoverlo. |

### Incidentes propuestos para la Actividad 8

1. **Aumento de latencia.** Introducir una demora controlada, observar la alerta,
   ejecutar el runbook y repetir la medición después de revertir el cambio.
2. **Cambio en los datos de entrada.** Comparar un lote de textos largos con
   la distribución de referencia. Si se detecta drift, revisar una muestra
   etiquetada para determinar si también se deterioró la calidad de las predicciones.
