# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 01:13:51 2019

@author: Sadil Khan
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score

# Importing Dataset
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv").values.astype('float32')
sample=pd.read_csv("sample_submission.csv")
test1=pd.read_csv("test.csv").values.astype('float32')

# Reshaping and Spliting
x=train.iloc[:,1:].values.astype('float32')
y=train.iloc[:,0].values

# Standarizing the Vector
x=abs(x-x.mean())/x.std()
test=abs(test-test.mean())/test.std()

"""#Reshaping
x=x.reshape(x.shape[0],28,28)
test=test1.reshape(test.shape[0],28,28)

#Plotting
for i in range(255,268):
    plt.subplot(272+(i-254))
    plt.imshow(test[i],cmap=plt.get_cmap('gray'))
    plt.title(predict[i])
    plt.show()"""


# Adding Dimension for color channel Grey
x=x.reshape(x.shape[0],28,28,1)
test=test.reshape(test.shape[0],28,28,1)

# One Hot Encoding
y=to_categorical(y,num_classes=10)

# Classifier
classifier=Sequential()

# Adding Convolutional Layer
classifier.add(Conv2D(32,3,3,input_shape=(28,28,1),activation='relu'))
classifier.add(Conv2D(32,3,3,activation='relu'))

# Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
# Adding Convolutional Layer
classifier.add(Conv2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

# Flatten'
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(units=300,activation='relu'))
classifier.add(Dense(units=124,activation='relu'))

classifier.add(Dense(units=10,activation='softmax'))

# Compiling The CNN
classifier.compile(optimizer=RMSprop(lr=0.0006),loss='categorical_crossentropy',metrics=['accuracy'])

# Spliting The Dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)

# Data Augmentation
datagen=ImageDataGenerator(rescale=1/.255,rotation_range=10,
                           shear_range=50,zoom_range=0.15,
                           height_shift_range=0.2,
                           width_shift_range=0.1)
datagen.fit(x_train)

# Fitting The Model
classifier.fit(x_train,y_train,batch_size=88,epochs=30)

# Accuracy Function
def accuracy():
    y_pred=classifier.predict_classes(x_test)
    y_test_set=[]
    for i in range(len(y_test)):
        count=0
        for j in range(10):
            if y_test[i][j]!=0:
                y_test_set.append(j)
            else:
                count=count+1
        if count==10:
            y_test_set.append(0)
    return(accuracy_score(y_test_set,y_pred))
accuracy()
# Predicting Actual Test Set
predict=classifier.predict_classes(test)

# Making A CSV File
sample["Label"]=predict
sample.to_csv("sample.csv",index=False)                      