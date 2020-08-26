from flask import Flask, render_template, request
import io
from keras.preprocessing import image
from keras.models import model_from_json
import json
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

with open('static/breeds.json') as f:
	dog_names = json.load(f)
	print(dog_names)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def extract_Xception(tensor):
	from keras.applications.xception import Xception, preprocess_input
	return Xception(weights='../saved_models/imagenet.h5', include_top=False).predict(preprocess_input(tensor))

def predict_breed_Xception(img_path): 	
	json_file = open('../saved_models/Xception.json', 'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	# load weights into new model
	model.load_weights("../saved_models/weights.best.Xception.hdf5")
	bottleneck_feature = extract_Xception(path_to_tensor(img_path))

	predicted_vector = model.predict(bottleneck_feature)

	predicted_index = np.argmax(predicted_vector)
	label = dog_names[predicted_index]
	return label, predicted_vector

@app.route('/')
def main():
    return render_template('app.html')

@app.route('/upload', methods = ['POST'])
def upload():
    file = request.files['file']

    filename = secure_filename(file.filename)
    path = os.path.join('static/', filename)
    file.save(path)
    label, vector = predict_breed_Xception(path)
    os.remove(path)

    res = dict(prediction = label)

    return json.dumps(res)
