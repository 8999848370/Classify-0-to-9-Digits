import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import glob
from keras import backend as k
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam 
from keras.models import Sequential
from keras.metrics import sparse_categorical_crossentropy
import cv2
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

def read(x,size):
    features =[]

    label = []

    for i in x:
        for j in dar:
            a1 = os.path.join("C:\\Users\\nilesh\\Desktop\\Applications\\Master\\Deeplearning\\DL project 2\\data\\"+i+"\\"+j+"/*g")
            for img in glob.glob(a1):
                image1 = cv2.imread(img)
                image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                im_resize = cv2.resize(image,(size,size), interpolation=cv2.INTER_CUBIC)
                features.append(im_resize)
                if j=="dogs":
                    label.append(0)
                else:
                    label.append(1)
    return np.array(features) , np.array(label)

path = ["train","test"]
dar = ["zero","one","two","three","four","five","six","seven","eight","nine"]
x_train , y_train = read(["train"],32)
x_test , y_test = read(["test"],32)
X_train=x_train.reshape(-1,32,32,1)
X_test =x_test.reshape(-1,32,32,1)

dataGen = ImageDataGenerator(width_shift_range = 0.1,
                            height_shift_range=0.1,
                            zoom_range = 0.2,
                            shear_range = 0.1,
                            rotation_range = 10)
# ImageDataGenrator is an image data augmentation technique that can be used to ARTIFICIALLY
# expand the size of a training dataset by creating modified versions of images in the datsset

# ImageDataGenrator is an image data augmentation technique that can be used to ARTIFICIALLY
# expand the size of a training dataset to improve the performance of the model to generalize.

dataGen.fit(X_train) # we want the genertor to know little bit about gthe dataset
                     # before we acually send it for training process

Y_train=to_categorical(y_train,2)
Y_test=to_categorical(y_test,2)
Y_test.shape

# this model is totaly based on Lenet model

no_of_filter = 60
size_of_filter1 = (5,5)
size_of_filter2 = (3,3)
size_of_pool  = (2,2)
noofNode = 500


# Convolution layer1 = filter=60 , filter = (5,5) 
# Convolution layer2 = filter=60 , filter = (5,5)

# Convolution layer3 = filter=60 , filter = (3,3)
# convolution layer4 = filter=60 , filter = (3,3)

# Dense layer 1 = node = 500
# Dense layer 2 = node = 10

model = Sequential()
model.add(Conv2D(60 , kernel_size = (5,5) ,input_shape = (32,32,1), activation = "relu"))
model.add(Conv2D(60 , kernel_size = (5,5) ,activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(60//2 , kernel_size = (3,3) ,activation = "relu"))
model.add(Conv2D(60//2 , kernel_size = (3,3) ,activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(500 , activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10 , activation = "softmax"))

model.compile(Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])

model.summary()

history = model.fit_generator(dataGen.flow(X_train , Y_train,
                                          batch_size = 50),
                                          steps_per_epoch = 2000,
                                          epochs = 10,
                                          validation_data = (X_test, Y_test))



import pickle
pickle.dump(model,open("model_trained.dat", "wb"))
pickle_out.close()

# Convolution layer1 = filter=60 , filter = (5,5) 
# Convolution layer2 = filter=60 , filter = (5,5)

# Convolution layer3 = filter=60 , filter = (3,3)
# convolution layer4 = filter=60 , filter = (3,3)

# Dense layer 1 = node = 500
# Dense layer 2 = node = 10

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 1
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

#### LOAD THE TRAINNED MODEL 

model = pickle.load( open("model_trained.dat","rb"))

#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)
    print(classIndex,probVal)

    if probVal> threshold:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break