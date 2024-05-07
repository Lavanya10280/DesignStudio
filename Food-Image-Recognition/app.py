import tensorflow
from flask import Flask, request, render_template
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


tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# define label meaning
label = ['apple pie',
         'baby back ribs',
         'baklava',
         'beef carpaccio',
         'beef tartare',
         'beet salad',
         'beignets',
         'bibimbap',
         'bread pudding',
         'breakfast burrito',
         'bruschetta',
         'caesar salad',
         'cannoli',
         'caprese salad',
         'carrot cake',
         'ceviche',
         'cheese plate',
         'cheesecake',
         'chicken curry',
         'chicken quesadilla',
         'chicken wings',
         'chocolate cake',
         'chocolate mousse',
         'churros',
         'clam chowder',
         'club sandwich',
         'crab cakes',
         'creme brulee',
         'croque madame',
         'cup cakes',
         'deviled eggs',
         'donuts',
         'dumplings',
         'edamame',
         'eggs benedict',
         'escargots',
         'falafel',
         'filet mignon',
         'fish and_chips',
         'foie gras',
         'french fries',
         'french onion soup',
         'french toast',
         'fried calamari',
         'fried rice',
         'frozen yogurt',
         'garlic bread',
         'gnocchi',
         'greek salad',
         'grilled cheese sandwich',
         'grilled salmon',
         'guacamole',
         'gyoza',
         'hamburger',
         'hot and sour soup',
         'hot dog',
         'huevos rancheros',
         'hummus',
         'ice cream',
         'lasagna',
         'lobster bisque',
         'lobster roll sandwich',
         'macaroni and cheese',
         'macarons',
         'miso soup',
         'mussels',
         'nachos',
         'omelette',
         'onion rings',
         'oysters',
         'pad thai',
         'paella',
         'pancakes',
         'panna cotta',
         'peking duck',
         'pho',
         'pizza',
         'pork chop',
         'poutine',
         'prime rib',
         'pulled pork sandwich',
         'ramen',
         'ravioli',
         'red velvet cake',
         'risotto',
         'samosa',
         'sashimi',
         'scallops',
         'seaweed salad',
         'shrimp and grits',
         'spaghetti bolognese',
         'spaghetti carbonara',
         'spring rolls',
         'steak',
         'strawberry shortcake',
         'sushi',
         'tacos',
         'octopus balls',
         'tiramisu',
         'tuna tartare',
         'waffles']

nu_link = 'https://www.nutritionix.com/food/'

# Loading the best saved model to make predictions.
base_model = InceptionV3(weights='imagenet', include_top=True)
model_best = Model(inputs=base_model.input, outputs=base_model.output)
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
            # Load pre-trained InceptionV3 model (excluding top classification layer)
            model = InceptionV3(weights='imagenet', include_top=False)
            
            # Preprocess your food image
            img_path = filename
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Get features from InceptionV3 model
            features = model.predict(x)
            
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

        # if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
        #     pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])

        
        pa['image'] = filename
        x = dict()
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                x[label] = "{:.2f}".format(score * 100)
        # x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
        # x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
        # x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))
        pa['result'] = x
        # pa['nutrition'] = 'needs to be filled'
        # pa['food'] = f'{nu_link}'
        # pa['idx'] = i - start[0]
        pa['quantity'] = 100

        pack[0].append(pa)
        passed[0] += 1

    start[0] = passed[0]
    print('successfully packed')
    # compute the average source of calories
    # for p in pack[0]:
    #     nutrients[0]['value'] = (nutrients[0]['value'] + p['nutrition'][0]['value'])
    #     nutrients[1]['value'] = (nutrients[1]['value'] + p['nutrition'][1]['value'])
    #     nutrients[2]['value'] = (nutrients[2]['value'] + p['nutrition'][2]['value'])
    #     nutrients[3]['value'] = (nutrients[3]['value'] + p['nutrition'][3]['value'])
    #     nutrients[4]['value'] = (nutrients[4]['value'] + p['nutrition'][4]['value'])

    # nutrients[0]['value'] = nutrients[0]['value'] / num[0]
    # nutrients[1]['value'] = nutrients[1]['value'] / num[0]
    # nutrients[2]['value'] = nutrients[2]['value'] / num[0]
    # nutrients[3]['value'] = nutrients[3]['value'] / num[0]
    # nutrients[4]['value'] = nutrients[4]['value'] / num[0]

    return render_template('results.html', pack=pack[0])



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
