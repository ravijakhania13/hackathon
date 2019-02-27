#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import tensorflow as tf
import json
import requests

df = pd.read_csv('Sonar.csv')
print(df.iloc[0].tolist())

train_data = df[df.columns[0:60]].values
train_labels = df[df.columns[60]]

eval_data = df[df.columns[0:60]].values
eval_labels = df[df.columns[60]]

data = json.dumps({"signature_name": "model", "instances": df.iloc[0].tolist()})

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/saved_model:predict', data=data, headers=headers)
print (json_response.text)

predictions = json.loads(json_response.text)['probabilities']

print ("Prediction: ", np.argmax(predictions[0]))


# In[ ]:




