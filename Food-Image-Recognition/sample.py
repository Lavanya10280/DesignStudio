from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import PIL
import sys

try:
    # Load pre-trained InceptionV3 model (excluding top classification layer)
    model = InceptionV3(weights='imagenet', include_top=False)

    # Preprocess your food images
    img_path = 'test_images/bread.jpg'
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
