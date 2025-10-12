from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64

# Initialize flask app
app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model('app/mnist_classification.h5')

# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('drawing.html')

# Handle POST request
@app.route('/', methods=['POST'])
def canvas():
    canvasdata = request.form['canvasimg']
    encoded_data = canvasdata.split(',')[1]

    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(gray_image)  # <--- important
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    gray_image = gray_image / 255.0  # normalize

    img = np.expand_dims(gray_image, axis=0)

    prediction = np.argmax(model.predict(img), axis=1)[0]
    print(f"Prediction Result: {prediction}")
    return render_template('drawing.html', response=str(prediction), canvasdata=canvasdata, success=True)