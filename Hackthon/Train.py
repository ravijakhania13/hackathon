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


# model_path = "./../output_data/model_output.ckpt"


# In[3]:


#define the one hot encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[4]:


#Read the sonar dataset
df = pd.read_csv('creditcard.csv')
# print(len(df.columns))
X = df[df.columns[0:30]].values
y = df[df.columns[30]]


# In[5]:


#encode the dependent variable containing categorical values
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)


# In[6]:


#Transform the data in training and testing
X,Y = shuffle(X,Y,random_state=1)
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=42)
train_x.shape[0]


# In[7]:


#define and initialize the variables to work with the tensors
learning_rate = 0.1
training_epochs = 50


# In[8]:


#Array to store cost obtained in each epoch
cost_history = np.empty(shape=[1],dtype=float)


# In[9]:


n_dim = X.shape[1]
n_class = 2
n_ans = 1


# In[10]:


x = tf.placeholder(tf.float32,[None,n_dim],name = 'x')
W = tf.Variable(tf.zeros([n_dim,n_class]),name = 'W')
b = tf.Variable(tf.zeros([n_class]),name='b')
Accuracy = tf.Variable(tf.zeros([1],dtype=tf.float32),name = 'Accuracy')
# print (Accuracy[0])
# Accuracy = tf.placeholder(tf.float32,[None,n_ans],name = "Accuracy")


# In[11]:


#initialize all variables.
init = tf.global_variables_initializer()


# In[12]:


#define the cost function
y_ = tf.placeholder(tf.float32,[None,n_class],name = "y_")
y = tf.nn.softmax(tf.matmul(x, W)+ b,name = "y")
# cost_function = tf.reduce_sum(tf.square(y_ - y))
cost_function = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)),reduction_indices=[1]))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


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
#     print('epoch : ', epoch,  ' - ', 'cost: ', cost)


# In[15]:


# print("Accuracy:",sess.run(Accuracy))


# In[16]:


pred_y = sess.run(y, feed_dict={x: test_x})


# In[17]:


#Calculate Accuracy
correct_prediction = tf.equal(tf.argmax(pred_y,1), tf.argmax(test_y,1))
# op = Accuracy.assign(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100)
# Accuracy.load(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100, sess)
# Accuracy.assign(Accuracy + (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100))
# print (sess.run(Accuracy.initializer))
# Accuracy = tf.Variable(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100,name = "Accuracy")
Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
# sess.run(op)
print("Accuracy:",sess.run(Accuracy))


# In[18]:


saver.save(sess, './model/train_data')
# sess.close()
# save_path = saver.save(sess,model_path)
# print("Model saved to file: %s " % save_path)


# In[19]:


# chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='W2', all_tensors=True)


# In[ ]:





# In[ ]:





# In[ ]:




