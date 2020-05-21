from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
#from matplotlib import pyplot as plt
from PIL import Image

cwd = os.getcwd()
output_model_path = os.path.join(cwd,"dogs_cats_classification_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict-dogs-cats-api',methods=["POST"])
def predict_api():
    data = request.get_json()
    print("data {}".format(data))
    image = request.files['Image']
    print("image {}".format(image))
    img = Image.open(image)
    Image_new = img
    img = np.array(img)
    img = cv2.resize(img,(80,80))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = img/255.0
    model = pickle.load(open(output_model_path,"rb"))
    img = np.array(img).reshape(-1,80,80,1)
    prediction = model.predict(img)
    predicted_val = [int(round(p[0])) for p in prediction]
    pre = predicted_val[0]
    if(pre==0):
        return "The picture is of a CAT!!!!!"
    elif(pre==1):
        return "The picture is of a DOG!!!!!"


@app.route('/predict-dog-cat-frontend',methods=["POST"])
def predict_front_end():
    data = request.form.get('filename')
    image = request.files['filename']
    img = Image.open(image)
    img = np.array(img)
    img = cv2.resize(img,dsize=(80,80))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = img/255.0
    model = pickle.load(open(output_model_path,"rb"))
    img = np.array(img).reshape(-1,80,80,1)
    prediction = model.predict(img)
    predicted_val = [int(round(p[0])) for p in prediction]
    pre = predicted_val[0]
    result = None
    if(pre==0):
        result =  "The picture is of a CAT!!!!!"
    elif(pre==1):
        result =  "The picture is of a DOG!!!!!"
    return render_template("index.html",prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
