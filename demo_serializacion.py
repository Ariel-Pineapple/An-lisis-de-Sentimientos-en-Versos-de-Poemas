import pickle

# Objeton sencilllo para ilustar la serialización.

confuguracion = {
    "modelo": "Poem Sentiment Analysis",
    "version": "1.0",
    "autor": "Tu Nombre",
    "Clases:":["Positivo", "Negativo", "Neutral"]

}



# Serialización del objeto a un archivo
with open("configuracion.pkl", "wb") as archivo:
    pickle.dump(confuguracion, archivo)



print("Objeto serializado y guardado en 'configuracion.pkl'.") 

# Deserialización del objeto desde el archivo

with open("configuracion.pkl", "rb") as archivo:
    configuracion_cargada = pickle.load(archivo)

print("Objeto deserializado desde 'configuracion.pkl':")    

