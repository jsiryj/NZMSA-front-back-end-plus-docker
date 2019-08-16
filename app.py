import base64 #for sending as url
import numpy as np 
import io
from PIL import Image
import keras
import os
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_our_model():
	global model # so we can access the model outside this function
	if os.path.exists("transfer.h5"):
		model = load_model("transfer.h5")
	else:
		print("File {} not found!".format("transfer.h5"))
	print("Model is successfully loaded")
	
def prep_image_for_class(image, required_size):
	if image.mode != "RGB":
		image = image.convert("RGB") # assure that we use RGB images
	image = image.resize(required_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	
	return image
	
print("Wait while model is loaded")
load_our_model()

@app.route("/")
def greet():
	return("Hello World")

@app.route("/classify", methods=["POST"])
def classify():
	comm = request.get_json(force=TRUE) # receive the API request
	encoded_comm = comm['image']
	decoded_comm = base64.b64decode(encoded_comm)
	image = Image.open(io.BytesIO(decoded_comm))
	prepared_image = prep_image_for_class(image, required_size=(224,224))
	
	classes = {0:'airplane', 1:'car', 2:'cat', 3:'dog', 4:'flower', 5:'fruit', 6:'motorbike', 7:'person'}
	
	prediction = classes[np.argmax(model.predict(prepared_image))]
	
	response = {
		'prediction': {
			'class': prediction[0]
		}
	}
	
	return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)