#### IMPORT ALL LIBRARIES
import os
import numpy as np 
import pandas as pd
import config
import json
import tensorflow
from tensorflow import keras
from keras.models import  Sequential
from keras.applications import MobileNetV2
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
from utilities import image_preprocessing
from utilities import CropSystem

### LOAD THE MODEL (.H5)
model = load_model(config.plant_model_path)

### DISEASE LABELS
with open(config.class_label_path, 'r') as file:
    class_labels = json.load(file)


app = Flask(__name__)

@app.route("/")
def Home_App():
    return render_template('index.html')

@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/plant_disease')
def plant_disease():
    return render_template('plant_disease.html')

############# PLANT DISEASE PREDICTION ##############
@app.route("/plant_disease_predict",methods=['POST','GET'])
def image_predict():
    if request.method=='POST':
        file = request.files['file']
        if file:
            # Save the file to a temporary location
            filename = "image.jpg"
            file.save(filename)
            # Preprocess Image
            img_array = image_preprocessing(filename)
            # Predict Image
            y_pred = model.predict(img_array)[0]
            y_pred = np.argmax(y_pred)

            result = f"The Given Plant Disease is : {class_labels['Label'][y_pred]}"

    return render_template('plant_disease.html',result=result)

############## CROP RECOMMENDATION SYSTEM ##############
@app.route("/crop_recommendation_predict", methods=['POST','GET'])
def predict_crop():
    if request.method=='POST':
        data = request.form
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        cropsystem = CropSystem(N,P,K,temperature,humidity,ph,rainfall)
        crop = cropsystem.recommended_crop()
        return render_template('result.html', result=crop)

    return render_template('crop_recommendation.html')



if __name__=="__main__":
    app.run(debug=True, port=config.PORT, host=config.HOST)

