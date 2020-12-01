#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ann assignment: handwritten digit classifier using neural network
# necessary modules being imported into the script
import matplotlib.pyplot as plt  # for plotting the image data as a graph
import matplotlib
import numpy as np  # module to use the builtin powerful mathematical and statiscal tools for python
# importing the necessary modules from keras: i.e 
# 1. layers: for computation in the neural network
# 2. model: built in artificial neural network framework to build our model
# 3.utils(utilities) : to transform data from one form/domain to another
from keras.utils import to_categorical
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Convolution2D, MaxPool2D, Flatten
from keras.models import Sequential
import PIL
import os
import tensorflow as tf
import pandas as pd
import random
rgb_weights = [0.2989, 0.5870, 0.1140]  # for grayscale conversion
training_images = []
test_images = []
training_tags = []
test_tags = []
batch_size = 5
img_height = 28
img_width = 28


# In[2]:


def image_flattening(path):
    '''this function is for feeding a well shaped image for model evaluation post training '''
    an_image = matplotlib.image.imread(path)
    plt.imshow(an_image, cmap=plt.get_cmap("gray"))
    # printing the test image for reference
    img2_data = an_image.reshape((-1,28,28,3))
    # reshaping the image for it be fed into the convolutional layer
    return img2_data


# In[3]:


filenames = []
for dire, path, filey in os.walk(r"C:\Users\User\Downloads\ann_paint_dataset-20201123T074433Z-001\ann_paint_dataset"):
    for ft in filey:
        filenames.append(ft)
categories = []
fi = r"C:\Users\User\Downloads\ann_paint_dataset-20201123T074433Z-001\ann_paint_dataset"
for i in range(10):
    for dire, pathy, fille1 in os.walk(os.path.join(fi,str(i))):
        for fty in fille1:
            if i == 0:
                categories.append(0)
            elif i == 1:
                categories.append(1)
            elif i == 2:
                categories.append(2)
            elif i == 3:
                categories.append(3)
            elif i == 4:
                categories.append(4)
            elif i == 5:
                categories.append(5)
            elif i == 6:
                categories.append(6)
            elif i == 7:
                categories.append(7)
            elif i == 8:
                categories.append(8)
            elif i == 9:
                categories.append(9)
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
#creating data labels for the images where each image is represented by the numbers contained in the image
# creating the pandas data frame of images and their respective labels(categories)


# In[4]:


df.head()
# printing the top most entries of the dataframe


# In[5]:


df.tail()
# printing the lower most entries of the data frame


# In[6]:


df['category'].value_counts().plot.bar()
# pictorial representation of the size of categories in the data set used


# In[7]:


image = load_img(r"C:\Users\User\Downloads\ann_paint_dataset-20201123T074433Z-001\ann_paint_dataset\7\_7_.jpg")
plt.imshow(image)
# printing a random image from the training dataset


# In[8]:


# using tensorflows built it function to generate dataset from images and splitting the 
# dataset into train and validation subparts
#1. generating train data set from input images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\User\Downloads\ann_paint_dataset-20201123T074433Z-001\ann_paint_dataset",
    label_mode='categorical',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
#2. generating validation(test) dataset from input images
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\User\Downloads\ann_paint_dataset-20201123T074433Z-001\ann_paint_dataset",
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# In[9]:


# building the neural network model:
# 1. the neural network will consist of 7 layers, 2  convolution layers 2 maxpool layers, flattening layer
# and 2 dense layers.
# 2. last layer with 10 neurons, each for a digit from 0-9.





# Initialize the CNN
classifier_model = Sequential()

# Add the Layers of the CNN

# Step 1 - Convolution : Applying several feature detectors 
# to the input image and create a feature map for each feature detector
# This step creates many feature maps to obtain each convolutional layer

# 32 =  Number of feauture maps constructed from 3x3 feature detectors
# 3,3 = dimension of each feature detector

classifier_model.add(Convolution2D(32,(3,3), input_shape = (28,28,3), activation = 'relu'))
# ReLu activation function: the activation function used is"rectified linear activation fuction" aka relu,
# the relu activation function is a linear piecewise function where the output is equal to input if 
# the input is postive else the output is 0.


# Step 2 - Pooling Layer
# Used to reduce the size of the feature maps
# i.e. it creates a smaller feature map
# It is used to reduce the number of nodes in the future fully connected layer
classifier_model.add(MaxPool2D(pool_size = (2,2)))


# Adding a second convolutional layer
classifier_model.add(Convolution2D(32,(3,3), activation = 'relu'))
# And a second pooling layer
classifier_model.add(MaxPool2D(pool_size = (2,2)))


# Step 3 - Flattening
# We are flatteting all the Pooling Layers into one huge vector that will be the input to the ANN
classifier_model.add(Flatten())

# Step 4 - Full Connection
classifier_model.add(Dense(64 , activation = 'relu' ))
# second dense layer
#classifier_model.add(Dense(64 , activation = 'relu' ))
# softmax as we have 10 possible outcomes 
classifier_model.add(Dense(10 , activation = 'softmax' ))
# Softmax activation function: is used to normalize the output of a network to a probablity distribution
# over the predicted output classes, is often used for the last layer activation function.

classifier_model.summary()

#Step 5 - Initialise the CNN
#classifier_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[10]:


# compiling the model of the above neural network; by using an optimizer and
# a loss function; a loss function is to evaluate the performance and indicate
# how far the predicted value is from actual value and this calculation can be 
# used by the optimiser to improve the neural networks weights

classifier_model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',  # loss function evaluation for 10 classes
    metrics = ['accuracy']
)


# In[11]:


# training the above built neural network
history = classifier_model.fit_generator(
    train_ds,
    epochs = 15, # the number of training iterations over the entire data set
    validation_data = val_ds,
    #batch_size = batch_size   # the number of input cases to be provided in each step of each iteration here first 10 samples of the data set will be fed until we ehaust the dataset
    )


# In[12]:


# evaluate the model over the test data set
hist = classifier_model.evaluate(
    val_ds,
)


# In[13]:


# plotting a graph representing the metrics of the model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper left')
plt.show()


# In[14]:


classifier_model.save(r"C:\Users\User\Downloads")
# saving the entire model


# In[15]:


prediction = classifier_model.predict(image_flattening(r"C:\Users\User\Downloads\6\6.jpeg"))
# the above line is  where an image of the size 28*28 pixels path can be provided for evaluation
print(prediction)  # the list of probabilities of the input being the respective number
print(np.argmax(prediction, axis = 1))
# the neuron with highest probability that is close to 1 is the output i.e. the prediciton of the 
# handwritten digit in the image provided


# In[ ]:




