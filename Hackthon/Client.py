#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
import json
import requests

df = pd.read_csv('Sonar.csv')

eval_data = df[df.columns[0:60]].values
eval_labels = df[df.columns[60]]

data = json.dumps({"signature_name": "model", "instances": eval_data[0].tolist()})

headers = {"content-type": "application/json"}
json_response = requests.post('http://10.2.133.28:8501/v1/models/sonar_model:predict', data=data, headers=headers)
print (json_response.text)

predictions = json.loads(json_response.text)

print ("Prediction: ", np.argmax(predictions[0]))
