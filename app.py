from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import Counter
import seaborn as sns


app = Flask(__name__, template_folder='template')

model = tf.keras.models.load_model("my_model.h5")

@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')

    image_array = np.array(image)
    resized_image = resize(image_array, (28, 28))
    flattened_image = resized_image.flatten()
    reshaped_image = flattened_image.reshape(1, 784)
    prediction = model.predict(reshaped_image)
    # Convert prediction to percentage
    confidence_percentage = prediction[0][0] * 100
    
    if confidence_percentage >= 50:
        prediction_result = '1'
    else:
        prediction_result = '0'

    # Format prediction as a string
    prediction_string = f'{confidence_percentage:.2f}%'

    return render_template('index.html', prediction=prediction_result, confidence=prediction_string)


if __name__ == '__main__':
    app.run(port=3245, debug=True)