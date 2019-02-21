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


model_path = "./../output_data/model_output"


# In[3]:


Path = sys.argv[1]
# Path = "./creditcard.csv"


# In[4]:


#define the one hot encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[5]:


#Read the sonar dataset
df = pd.read_csv(Path)
# print(len(df.columns))
X = df[df.columns[0:30]].values
y = df[df.columns[30]]


# In[6]:


#encode the dependent variable containing categorical values
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


# x = tf.get_variable("x", shape=[X.shape[0],n_dim], initializer = tf.zeros_initializer())
# x = tf.placeholder(tf.float32,[None,n_dim],name = 'x')
W = tf.get_variable("W", shape=[n_dim,n_class], initializer = tf.zeros_initializer())
b = tf.get_variable("b", shape=[n_class], initializer = tf.zeros_initializer())
# Accuracy = tf.get_variable("Accuracy", shape=[1], initializer = tf.zeros_initializer())
# print (Accuracy[0])
# Accuracy = tf.placeholder(tf.float32,[None,n_ans],name = "Accuracy")


# In[12]:


saver = tf.train.Saver()


# In[13]:


x = tf.placeholder(tf.float32,[None,n_dim],name = 'x')
Accuracy = tf.placeholder(tf.float32,[None,1],name = 'Accuracy')
y_ = tf.placeholder(tf.float32,[None,n_class],name = "y_")
y = tf.nn.softmax(tf.matmul(x, W)+ b,name = "y")


# In[14]:


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model/train_data.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
#     saver.restore(sess, "/tmp/h/model.ckpt")
#     print("Model restored.")
    # Check the values of the variables
#     print(W2.eval())
#     test = fashion_mnist.test
    pred_y = sess.run(y, feed_dict={x: X})
    correct_prediction = tf.equal(tf.argmax(pred_y,1), tf.argmax(Y,1))
    Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
# sess.run(op)
#     print (pred_y)
    print("Accuracy:",sess.run(Accuracy))
#     (n_x, m) = test.images.T.shape
#     n_y = test.labels.T.shape[0]
#     X, Y = create_placeholders(n_x, n_y)
#     Z3 = forward_propagation(X,W1,W2,W3,b1,b2,b3)
    
#     correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
#     print ("Test Accuracy:", accuracy.eval({X: test.images.T, Y: test.labels.T}))


# In[ ]:





# In[ ]:




