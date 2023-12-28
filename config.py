# It contains path of model,features, pickle file

import os
## For Plant Disease
plant_model_path = os.path.join("artifacts", "plant_disease_model.h5")
class_label_path = os.path.join("artifacts","disease_labels.json")

## For Crop Recommendation
crop_model_path = os.path.join('artifacts', 'new_rf_model.pickle')
features_path = os.path.join('artifacts',  'features_data.json')
labels_path = os.path.join('artifacts',  'crop_labels.json')


### PORT AND HOST NUMBER
PORT = 8080
HOST = "0.0.0.0"
