from os import listdir
from os.path import isfile, join
import cv2
import numpy as  np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import numpy
from keras.optimizers import Adam
OPTIMIZER = Adam(lr=0.0001, decay=8e-9)

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense

print(cv2.__version__)
mypath='train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Define model
model = Sequential()
model.add(Dense(6000, input_dim=8192, activation= "relu"))
model.add(Dense(4000, activation= "relu"))
model.add(Dense(2000, activation= "relu"))
model.add(Dense(4000, activation= "relu"))
model.add(Dense(6000, activation= "relu"))
model.add(Dense(8192, activation= "relu"))
#model.add(Dense(1))


print(len(onlyfiles))
a=np.zeros((len(onlyfiles),4096), dtype=float)
x=np.zeros((len(onlyfiles),8192), dtype=float)
test=np.zeros((len(onlyfiles),8192), dtype=float)
dis=np.zeros(len(onlyfiles), dtype=float)

#a=[]
c=0
for i in (onlyfiles):
    print(c)
    path=mypath+'/'+i
    temp=cv2.imread( path, 0)
    temp = np.reshape(temp, 4096)
    a[c]=temp
    c=c+1
    #a.append(temp)

for i in range(0,len(a)):
    x[i]=np.concatenate((a[10], a[i]), axis=0)


x=x/255
model.compile(loss= "mean_squared_error" , optimizer=OPTIMIZER, metrics=["mean_squared_error"])
model.fit(x, x, epochs=3)


for i in range(0,len(a)):
    test[i]=np.concatenate((a[i], a[10]), axis=0)
test=test/255
res=model.predict(test)


for i in range(0,len(a)):
    dis[i] = numpy.linalg.norm(res[i] - test[i])

sor=numpy.argsort(dis)
sorti=numpy.sort(dis)
c=0
for i in range(0,500):
    if (sor[i]<500):
        c=c+1
print(c)
print(sorti)

