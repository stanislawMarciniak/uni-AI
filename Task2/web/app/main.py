from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64

# Initialize flask app
app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model("app/mnist_classification.h5")


# Handle GET request
@app.route("/", methods=["GET"])
def drawing():
    return render_template("drawing.html")


# Handle POST request
@app.route("/", methods=["POST"])
def canvas():
    # Recieve base64 data from the user form
    canvasdata = request.form["canvasimg"]
    encoded_data = request.form["canvasimg"].split(",")[1]

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Center the digit in the frame (MNIST digits are centered)
    # Find bounding box of digit
    coords = cv2.findNonZero(gray_image.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        centered = np.zeros((28, 28), dtype=np.float32)
        # Calculate centering offsets
        digit = gray_image[y : y + h, x : x + w]
        aspect = w / h
        if aspect > 1:
            new_w = 20
            new_h = int(20 / aspect)
        else:
            new_h = 20
            new_w = int(20 * aspect)

        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Center in 28x28
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        centered[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = digit
        gray_image = centered

    # Expand dimensions
    img = np.expand_dims(gray_image, axis=0)

    try:
        np.set_printoptions(linewidth=1000)
        print(img)
        prediction = np.argmax(model.predict(img))
        print(f"Prediction Result : {str(prediction)}")
        return render_template(
            "drawing.html",
            response=str(prediction),
            canvasdata=canvasdata,
            success=True,
        )
    except Exception as e:
        return render_template("drawing.html", response=str(e), canvasdata=canvasdata)
