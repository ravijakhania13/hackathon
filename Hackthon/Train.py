#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
import math as mt
import sys
eps = np.finfo(float).eps
from tensorflow.python.tools import inspect_checkpoint as chkp


# In[2]:


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[3]:


df = pd.read_csv('Sonar.csv')
# print(len(df.columns))
X = df[df.columns[0:60]].values
y = df[df.columns[60]]


# In[4]:


encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)


# In[5]:


X,Y = shuffle(X,Y,random_state=1)
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=42)
train_x.shape[0]


# In[6]:


#define and initialize the variables to work with the tensors
learning_rate = 0.1
training_epochs = 50


# In[7]:


#Array to store cost obtained in each epoch
cost_history = np.empty(shape=[1],dtype=float)


# In[8]:


n_dim = X.shape[1]
n_class = 2
n_ans = 1


# In[9]:


x = tf.placeholder(tf.float32,[None,n_dim],name ="x")
y_ = tf.placeholder(tf.float32,[None,n_class],name ="y_")
# W = tf.Variable(tf.zeros([n_dim,n_class]),name = 'W')
# b = tf.Variable(tf.zeros([n_class]),name='b')
W = tf.get_variable("W",[n_dim,n_class],initializer=tf.zeros_initializer())
b = tf.get_variable("b",[n_class],initializer=tf.zeros_initializer())
Accuracy = tf.Variable(tf.zeros([1],dtype=tf.float32),name = 'Accuracy')


# In[10]:


#initialize all variables.
init = tf.global_variables_initializer()


# In[11]:


#define the cost function
y = tf.nn.softmax(tf.matmul(x, W)+ b,name = "y")
cost_function = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)),reduction_indices=[1]))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


# In[12]:


#TensorServing model

# tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

# export_path = "./model/1" + str(FLAGS.model_version)
export_path = "./model/1"
print("Exporting trained model to ", export_path)

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(y)


# In[13]:


#initialize the session
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
mse_history = []


# In[14]:


#calculate the cost for each epoch
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x,y_: train_y})
    cost_history = np.append(cost_history,cost)


# In[15]:


pred_y = sess.run(y, feed_dict={x: test_x})


# In[16]:


#Calculate Accuracy
correct_prediction = tf.equal(tf.argmax(pred_y,1), tf.argmax(test_y,1))
Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

print("Accuracy:",sess.run(Accuracy))


# In[18]:



prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
    inputs={"x": tensor_info_x},
    outputs={"y": tensor_info_y},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.TRAINING],
    signature_def_map={"model": prediction_signature,},saver=saver)

builder.save()
