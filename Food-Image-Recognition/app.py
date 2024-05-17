import tensorflow
from flask import Flask, request, render_template, jsonify
import csv
import math
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import PIL
import sys
import requests
import json


tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the top classification layer (softmax layer)
top_model = InceptionV3(weights='imagenet')
print('InceptionV3 model successfully loaded!')

start = [0]
passed = [0]
pack = [[]]
num = [0]

nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open('nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            name = ''
            continue
        else:
            name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]



@app.route('/')
def index():
    img = 'static/profile.jpg'
    return render_template('index.html', img=img)


@app.route('/recognize')
def magic():
    return render_template('recognize.html', img=file)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.getlist("img")
    for f in file:
        filename = secure_filename(str(num[0] + 500) + '.jpg')
        num[0] += 1
        name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('save name', name)
        f.save(name)

    pack[0] = []
    return render_template('recognize.html', img=filename)


@app.route('/predict')
def predict():
    result = []
    # pack = []
    print('total image', num[0])
    for i in range(start[0], num[0]):
        pa = dict()

        filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
        print('image filepath', filename)
        
        try:
            # Preprocess your food image
            img_path = filename
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Load the top classification layer (softmax layer)
            top_model = InceptionV3(weights='imagenet')

            # Predict classes for your food image
            predictions = top_model.predict(x)

            # Decode the predictions to get human-readable labels
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            # Print the top predicted classes and their probabilities
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                print(f"{i + 1}: {label} ({score:.2f})")

        except PIL.UnidentifiedImageError as e:
                print(f"Error: Unable to identify image file '{img_path}'")
                sys.exit(1)

        
        pa['image'] = filename
        x = dict()
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                x[label] = "{:.2f}".format(score * 100)
                if i == 0:
                    nutrition_info = get_nutrition_info(label)
       
        pa['result'] = x
        pa['quantity'] = 100
        pa['nutrition info'] = nutrition_info
        pack[0].append(pa)
        passed[0] += 1

    start[0] = passed[0]
    print('successfully packed')

    return render_template('results.html', pack=pack[0], nutrition_info=nutrition_info)


def get_nutrition_info(food_item):

    # Define your application ID and application key
    app_id = '5f26746b'
    app_key = '1136c06fa0a881845197eaf0ddab841d'

    # Define the headers with authentication information
    headers = {
        'Content-Type': 'application/json',
        'x-app-id': app_id,
        'x-app-key': app_key
    }

    # Make a GET request to the Nutritionix API to search for the food item
    response = requests.post('https://trackapi.nutritionix.com/v2/natural/nutrients',
                            headers=headers,
                            json={'query': food_item}
                            )

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("Nutrition Info Retrieved")
        # Return the JSON response from the Nutritionix API
        return response.json()
    else:
        # Return an error message if the request failed
        print("Nutrition Info Not Retrieved")
        return {'error': 'Failed to fetch nutrition data'}


@app.route('/update', methods=['POST'])
def update():
    return render_template('index.html', img='static/P2.jpg')


if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using
            python server.py
        Show the help text using
            python server.py --help
        """
        HOST, PORT = host, port
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    run()

