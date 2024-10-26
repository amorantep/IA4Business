from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS

# Cargar el modelo guardado
with open("modelo_power_plant.pkl", "rb") as f:
    model = pickle.load(f)

# Ruta para servir el formulario HTML
@app.route("/")
def home():
    return render_template("power_plant_form.html")

# Ruta para hacer la predicción
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Obtener las características de entrada
    input_features = np.array([
        data["AT"],
        data["V"],
        data["AP"],
        data["RH"]
    ]).reshape(1, -1)  # Asegurar que sea una matriz 2D

    # Hacer la predicción
    prediction = model.predict(input_features)

    # Retornar la predicción como JSON
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
