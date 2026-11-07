import streamlit as st


st.title("Poem Sentiment Analysis")
st.success("This is a demo of the Poem Sentiment Analysis model. You can input a poem and the model will predict its sentiment.")



nombre = st.text_input("Escribe un poema:")


if st.button("Analizar"):
    if nombre:
        st.write("Poema ingresado:", nombre)
        # Aquí iría la lógica para analizar el sentimiento del poema
        # Por ejemplo, podrías llamar a una función que procese el texto y devuelva el sentimiento
        # sentiment = analyze_sentiment(nombre)
        # st.write("Sentimiento predicho:", sentiment)
    else:
        st.warning("Por favor, ingresa un poema para analizar.")


