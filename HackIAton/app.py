from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS

# Cargar el modelo desde el archivo
with open("modelo_casas.pkl", "rb") as f:
    model = pickle.load(f)

# Ruta principal para servir el HTML
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para hacer predicciones
@app.route("/predict", methods=["POST"])
def predict():
    # Obtener los datos enviados desde el frontend en formato JSON
    data = request.get_json()

    # Convertir los datos en un array para que el modelo los procese
    input_features = np.array([
        data["bedrooms"],
        data["bathrooms"],
        data["sqft_living"],
        data["sqft_lot"],
        data["floors"],
        data["waterfront"],
        data["view"],
        data["condition"],
        data["grade"],
        data["sqft_above"],
        data["sqft_basement"],
        data["yr_built"],
        data["yr_renovated"],
        data["zipcode"]
    ]).reshape(1, -1)  # Asegurar que sea una matriz 2D

    # Realizar la predicción
    prediction = model.predict(input_features)

    # Enviar la predicción como respuesta JSON
    return jsonify({"prediction": prediction[0]})

# Iniciar el servidor
if __name__ == "__main__":
    app.run(debug=True)
