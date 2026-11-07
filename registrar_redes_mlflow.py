import json
from pathlib import Path

import mlflow

BASE = Path(__file__).resolve().parent

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Analisis de sentimientos en versos")

modelos = {
    "BiLSTM": "modelo_bilstm",
    "DistilBERT": "modelo_distilbert",
}

for nombre, carpeta in modelos.items():
    archivo = BASE / "artifacts" / carpeta / "resultados.json"
    datos = json.loads(archivo.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name=nombre):
        mlflow.set_tag("origen", "Resultados previamente calculados")
        mlflow.log_param("modelo", nombre)
        mlflow.log_param(
            "dataset", "google-research-datasets/poem_sentiment"
        )

        # Conserva la configuración y las versiones del entrenamiento.
        for clave, valor in datos.items():
            if clave.startswith(("f1_", "reporte_")):
                continue
            if clave == "prueba_evaluada":
                continue

            if isinstance(valor, (dict, list)):
                valor = json.dumps(valor, ensure_ascii=False)

            mlflow.log_param(clave, valor)

        # Usa los mismos nombres de métricas para comparar modelos.
        for etapa in ("validacion", "prueba"):
            mlflow.log_metric(
                f"f1_macro_{etapa}",
                datos[f"f1_macro_{etapa}"],
            )

            reporte = datos[f"reporte_{etapa}"]
            mlflow.log_metric(
                f"accuracy_{etapa}", reporte["accuracy"]
            )

            for clase in ("negative", "positive", "no_impact"):
                for metrica in ("precision", "recall", "f1-score"):
                    nombre_metrica = metrica.replace("-", "_")
                    mlflow.log_metric(
                        f"{etapa}_{clase}_{nombre_metrica}",
                        reporte[clase][metrica],
                    )

        mlflow.log_artifact(
            str(archivo), artifact_path="evaluacion"
        )

    print(f"{nombre} registrado correctamente.", flush=True)

print("Registro terminado.")