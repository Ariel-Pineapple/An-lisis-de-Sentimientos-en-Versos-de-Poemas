from pathlib import Path
import json
import re

import joblib
import pandas as pd
import streamlit as st



BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"

NOMBRES = {
    0: "Negativo",
    1: "Positivo",
    2: "Sin impacto afectivo",
}

CARPETAS = {
    "Modelo clásico": "modelo_clasico",
    "BiLSTM": "modelo_bilstm",
    "DistilBERT": "modelo_distilbert",
}


def tokenizar(texto):
    # Misma tokenización utilizada en el entrenamiento de la BiLSTM.
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", texto.lower())



@st.cache_resource
def cargar_modelo(nombre):
    carpeta = ARTIFACTS / CARPETAS[nombre]

    # CPU para la primera versión de inferencia.
    if nombre == "Modelo clásico":
        return joblib.load(carpeta / "pipeline.joblib")

    if nombre == "BiLSTM":
        checkpoint = torch.load(
            carpeta / "modelo.pt",
            map_location="cpu",
            weights_only=True,
        )
        modelo = BiLSTMClasificador(checkpoint)
        modelo.load_state_dict(checkpoint["state_dict"])
        modelo.eval()
        return modelo, checkpoint

    tokenizador = AutoTokenizer.from_pretrained(
        carpeta, local_files_only=True
    )
    modelo = AutoModelForSequenceClassification.from_pretrained(
        carpeta, local_files_only=True
    )
    modelo.eval()
    return modelo, tokenizador


def predecir(nombre, texto):
    recurso = cargar_modelo(nombre)

    if nombre == "Modelo clásico":
        etiqueta = int(recurso.predict([texto])[0])
        return etiqueta, False

    if nombre == "BiLSTM":
        modelo, checkpoint = recurso
        tokens = tokenizar(texto)
        truncado = len(tokens) > checkpoint["max_len"]

        ids = [
            checkpoint["vocabulario"].get(token, checkpoint["unk_id"])
            for token in tokens[:checkpoint["max_len"]]
        ] or [checkpoint["unk_id"]]

        longitud = len(ids)
        ids += [checkpoint["pad_id"]] * (
            checkpoint["max_len"] - longitud
        )

        with torch.inference_mode():
            logits = modelo(
                torch.tensor([ids], dtype=torch.long),
                torch.tensor([longitud], dtype=torch.long),
            )

    else:
        modelo, tokenizador = recurso
        total_tokens = len(
            tokenizador(texto, truncation=False)["input_ids"]
        )
        truncado = total_tokens > 64

        entradas = tokenizador(
            texto,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length",
        )

        entradas = {
            clave: entradas[clave]
            for clave in ("input_ids", "attention_mask")
        }

        with torch.inference_mode():
            logits = modelo(**entradas).logits

    return int(logits.argmax(dim=1).item()), truncado


st.set_page_config(
    page_title="Sentimientos en versos",
    page_icon="📖",
    layout="wide",
)

st.title("Análisis de sentimientos en versos")
st.caption(
    "Luis Ariel Vázquez Piña · Proyecto de Inteligencia Artificial · Instituto Politécnico Nacional - CIC"
)

tab_analisis, tab_poema, tab_resultados, tab_metodo = st.tabs([
    "Analizar un verso",
    "Analizar un poema",
    "Resultados del experimento",
    "Metodología",
])

with tab_analisis:
    st.write(
        "Introduce un verso en inglés y compara las predicciones "
        "de los modelos entrenados."
    )

    with st.form("analizar"):
        texto = st.text_area(
            "Verso en inglés",
            value="The sun shines with hope",
            max_chars=2000,
            height=110,
        )
        seleccionados = st.multiselect(
            "Modelos",
            list(CARPETAS),
            default=list(CARPETAS),
        )
        ejecutar = st.form_submit_button("Analizar verso")

    if ejecutar:
        if not texto.strip():
            st.warning("Escribe un verso.")
        elif not seleccionados:
            st.warning("Selecciona al menos un modelo.")
        else:
            filas = []

            for nombre in seleccionados:
                try:
                    with st.spinner(f"Analizando con {nombre}..."):
                        etiqueta, truncado = predecir(
                            nombre, texto.strip()
                        )

                    filas.append({
                        "Modelo": nombre,
                        "Predicción": NOMBRES[etiqueta],
                        "Entrada truncada": "Sí" if truncado else "No",
                    })
                except Exception as error:
                    st.error(f"No se pudo ejecutar {nombre}: {error}")

                st.dataframe(
                    pd.DataFrame(filas),
                    hide_index=True,
                    use_container_width=True,
                )

    st.info(
        "El alcance evaluado es la clasificación de versos en inglés. "
        "La predicción puede diferir de la interpretación de un lector."
    )

with tab_resultados:
    filas = []

    for nombre, carpeta in CARPETAS.items():
        ruta = ARTIFACTS / carpeta / "resultados.json"

        if not ruta.exists():
            st.warning(f"Falta el registro de {nombre}.")
            continue

        with open(ruta, encoding="utf-8") as archivo:
            registro = json.load(archivo)

        filas.append({
            "Modelo": nombre,
            "F1 validación": registro["f1_macro_validacion"],
            "F1 prueba": registro.get("f1_macro_prueba"),
            "Accuracy prueba": registro.get(
                "reporte_prueba", {}
            ).get("accuracy"),
        })

    if filas:
        tabla = pd.DataFrame(filas)
        st.dataframe(
            tabla.style.format({
                "F1 validación": "{:.4f}",
                "F1 prueba": "{:.4f}",
                "Accuracy prueba": "{:.2%}",
            }, na_rep="Pendiente"),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Resultados de modelos guardados. Analizar un verso "
        "no recalcula estas métricas."
    )

with tab_metodo:
    st.markdown("""
    **Datos:** Poem Sentiment de Google Research.

    **Clases:** negativo, positivo y sin impacto afectivo (`no_impact`).
    Se excluyeron del experimento los 49 ejemplos `mixed`.

    **Particiones:** 843 versos de entrenamiento, 105 de validación
    y 104 de prueba.

    **Modelos:** TF–IDF con regresión logística, BiLSTM y DistilBERT.

    **Evaluación:** F1 macro como métrica principal, acompañado de
    métricas por clase y matrices de confusión.

    **Selección:** parámetros del modelo clásico mediante validación
    cruzada en entrenamiento; mejores épocas de las redes según
    F1 macro de validación.

    **Limitaciones:** corpus pequeño, clases desbalanceadas y versos
    aislados que pueden requerir contexto. Los resultados corresponden
    a una ejecución por modelo; no describen su variabilidad entre semillas.
    """)


with tab_poema:
    st.subheader("Análisis verso a verso")
    st.write(
        "Pega un poema en inglés, con un verso por línea. "
        "Cada verso se analiza de forma independiente."
    )

    with st.form("analizar_poema"):
        poema = st.text_area(
            "Poema en inglés",
            height=220,
            max_chars=20000,
            placeholder=(
                "The sun shines with hope\n"
                "My soul cries in silence\n"
                "The river runs beneath the bridge"
            ),
        )

        modelos_poema = st.multiselect(
            "Modelos para analizar el poema",
            list(CARPETAS),
            default=["Modelo clásico"],
        )

        enviar_poema = st.form_submit_button("Analizar poema")

    if enviar_poema:
        # Conservar el número de línea original y omitir líneas vacías.
        versos = [
            (numero, linea.strip())
            for numero, linea in enumerate(poema.splitlines(), start=1)
            if linea.strip()
        ]

        # Evitar mostrar resultados anteriores tras una nueva solicitud.
        st.session_state.pop("resultado_poema", None)

        if not versos:
            st.warning("Introduce al menos un verso.")
        elif not modelos_poema:
            st.warning("Selecciona al menos un modelo.")
        elif len(versos) > 100:
            st.warning(
                "Esta versión admite hasta 100 versos por análisis."
            )
        else:
            filas = []
            fallos = []
            total = len(versos) * len(modelos_poema)
            completados = 0
            progreso = st.progress(0.0)

            for nombre in modelos_poema:
                try:
                    # Cargar una vez; las predicciones reutilizan la caché.
                    cargar_modelo(nombre)

                    filas_modelo = []

                    for numero, verso in versos:
                        etiqueta, truncado = predecir(nombre, verso)

                        filas_modelo.append({
                            "Línea": numero,
                            "Verso": verso,
                            "Modelo": nombre,
                            "Sentimiento": NOMBRES[etiqueta],
                            "Entrada truncada": (
                                "Sí" if truncado else "No"
                            ),
                        })

                        completados += 1
                        progreso.progress(completados / total)

                    filas.extend(filas_modelo)

                except Exception as error:
                    fallos.append(
                        f"No se completó {nombre}: {error}"
                    )

            progreso.empty()

            if filas:
                resultado = pd.DataFrame(filas).sort_values(
                    ["Línea", "Modelo"]
                )

                st.session_state["resultado_poema"] = {
                    "tabla": resultado,
                    "fallos": fallos,
                }
            else:
                for fallo in fallos:
                    st.error(fallo)

    # Conservar resultados al descargar o interactuar con la página.
    guardado = st.session_state.get("resultado_poema")

    if guardado is not None:
        resultado = guardado["tabla"]

        for fallo in guardado["fallos"]:
            st.warning(fallo)

        st.dataframe(
            resultado,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Descargar predicciones en CSV",
            data=resultado.to_csv(index=False).encode("utf-8-sig"),
            file_name="sentimientos_por_verso.csv",
            mime="text/csv",
        )

        st.subheader("Distribución de predicciones")

        resumen = pd.crosstab(
            resultado["Modelo"],
            resultado["Sentimiento"],
        ).reindex(
            columns=list(NOMBRES.values()),
            fill_value=0,
        )

        st.bar_chart(resumen)

    st.caption(
        "La gráfica cuenta las predicciones por clase. "
        "No representa una evaluación de exactitud ni determina "
        "el sentimiento global del poema. Cada verso se procesa "
        "sin el contexto de las demás líneas."
    )


with tab_resultados:
    st.subheader("Comparación en prueba")

    if filas:
        # Reconstruir desde los registros para evitar depender
        # de variables usadas en otras pestañas.
        registros = []

        for nombre, carpeta in CARPETAS.items():
            ruta = ARTIFACTS / carpeta / "resultados.json"

            if ruta.exists():
                with open(ruta, encoding="utf-8") as archivo:
                    registro = json.load(archivo)

                if "f1_macro_prueba" in registro:
                    registros.append({
                        "Modelo": nombre,
                        "F1 macro": registro["f1_macro_prueba"],
                    })

        if registros:
            st.bar_chart(
                pd.DataFrame(registros).set_index("Modelo")
            )

    st.subheader("Matrices de confusión")

    modelo_elegido = st.selectbox(
        "Selecciona un modelo",
        list(CARPETAS),
        key="modelo_matriz",
    )

    ruta = (
        ARTIFACTS
        / CARPETAS[modelo_elegido]
        / "predicciones_test.csv"
    )

    if ruta.exists():
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        predicciones = pd.read_csv(ruta)

        fig, ax = plt.subplots(figsize=(6, 4))

        ConfusionMatrixDisplay.from_predictions(
            predicciones["label"],
            predicciones["prediccion"],
            labels=list(NOMBRES),
            display_labels=list(NOMBRES.values()),
            cmap="Blues",
            values_format="d",
            ax=ax,
        )

        ax.set_title(f"{modelo_elegido} — Prueba")
        ax.set_xlabel("Sentimiento predicho")
        ax.set_ylabel("Sentimiento real")
        plt.xticks(rotation=15)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Ejemplos de errores")

        errores = predicciones.loc[
            predicciones["label"] != predicciones["prediccion"]
        ].copy()

        errores["Real"] = errores["label"].map(NOMBRES)
        errores["Predicho"] = errores["prediccion"].map(NOMBRES)

        st.dataframe(
            errores[["verse_text", "Real", "Predicho"]].rename(
                columns={"verse_text": "Verso"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("No se encontró el archivo de predicciones de prueba.")