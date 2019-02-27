#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the required libraries
import matplotlib.pyplot as plt
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


# model_path = "./../output_data/model_output"


# In[3]:


# = sys.argv[1]
Path = "./Sonar.csv"


# In[4]:


#define the one hot encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[5]:


df = pd.read_csv(Path)
# print(len(df.columns))
X = df[df.columns[0:60]].values
y = df[df.columns[60]]


# In[6]:


encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)


# In[7]:


X,Y = shuffle(X,Y,random_state=1)
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=42)
train_x.shape[0]


# In[8]:


#define and initialize the variables to work with the tensors
learning_rate = 0.1
training_epochs = 1000


# In[9]:


#Array to store cost obtained in each epoch
cost_history = np.empty(shape=[1],dtype=float)


# In[10]:


n_dim = X.shape[1]
n_class = 2
n_ans = 1


# In[11]:


W = tf.get_variable("W", shape=[n_dim,n_class], initializer = tf.zeros_initializer())
b = tf.get_variable("b", shape=[n_class], initializer = tf.zeros_initializer())


# In[13]:


x = tf.placeholder(tf.float32,[None,n_dim],name = 'x')
Accuracy = tf.placeholder(tf.float32,[None,1],name = 'Accuracy')
y_ = tf.placeholder(tf.float32,[None,n_class],name = "y_")
y = tf.nn.softmax(tf.matmul(x, W)+ b,name = "y")


# In[15]:


export_path = "./model/1"

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_path)
    graph = tf.get_default_graph()

    pred_y = sess.run("y:0", feed_dict={"x:0": X})
    correct_prediction = tf.equal(tf.argmax(pred_y,1), tf.argmax(Y,1))
    Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    print("Accuracy:",sess.run(Accuracy))



# In[ ]:
