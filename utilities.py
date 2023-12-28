import os
import json
import numpy as np 
import pandas as pd 
import config 
import tensorflow
from tensorflow import keras
from keras.models import  Sequential
from keras.applications import MobileNetV2
from keras.models import load_model
from tensorflow.keras.preprocessing import image

### IMAGE PREPROCESSING
def image_preprocessing(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Scale The Data
    img_array = img_array/255

    return img_array


################################################################################################################################
### FOR CROP RECOMMENDATION PROJECT

import numpy as np
import pandas as pd  
import pickle
import json
import config
import joblib
import warnings
warnings.filterwarnings('ignore')


class CropSystem():
    def __init__(self,N,P,K,temperature,humidity,ph,rainfall):
        self.N = N          
        self.P = P           
        self.K = K           
        self.temperature = temperature  
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall

    def load_data(self):
        with open(config.crop_model_path, 'rb') as file:
            self.model = pickle.load(file)
        #self.model = joblib.load(config.model_path)

        with open(config.features_path, 'r') as file1:
            self.feature_data = json.load(file1)

        with open(config.labels_path, 'r') as file2:
            self.labels = json.load(file2)

    def recommended_crop(self):
        self.load_data()

        test_series = pd.Series(np.zeros(len(self.feature_data['columns'])),index=self.feature_data['columns'])

        test_series['N'] = self.N
        test_series['P'] = self.P
        test_series['K'] = self.K
        test_series['temperature'] = self.temperature
        test_series['humidity'] = self.humidity
        test_series['ph'] = self.ph
        test_series['rainfall'] = self.rainfall
        #print(test_series)
        output = self.model.predict([test_series])[0]
        #print(output)
        output = f"Recommended Crop For You : {self.labels['class_labels'][output]}"

        return output
    
