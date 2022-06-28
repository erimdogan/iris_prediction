import numpy as np
from flask import Flask, render_template, request, jsonify,redirect,url_for
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model("/home/erim/Desktop/python/model/iris.h5")
class_names = ['setosa', 'versicolor', 'virginica']
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predictApp():

    sepalLenght = request.form.get("sepalLenght")
    sepalWidth = request.form.get("sepalWidth")
    petalLenght = request.form.get("petalLenght")
    petalWidth = request.form.get("petalWidth")

    real_values = np.array([[sepalLenght,sepalWidth,petalLenght,petalWidth]]).astype(np.float32)

    predicted = model.predict(real_values)
    score=100*np.max(predicted[0])
    predicted_class = class_names[np.argmax(predicted)]
    
    return render_template("index.html", setosa = class_names[0], versicolor = class_names[1], virginica = class_names[2], accuracy = round(score,2), predicted = predicted_class)            
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)