
#included packages and libraries 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU,  MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Lambda
from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import numpy as np
import csv
import cv2
import os



#Hyper parameters
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 5 #since each epoch is taking appx. 150 secs and there is no significant improvement in loss after 5 epochs, I have set this value to 5
correction=0.20 #This correction factor will be added and subtracted from the steering angle of left and right camera respectively as a compensation to the different angles perceived by them.

#paths of all source folders that contain dataset
PATH = [
         "1/1/",
         "data1/data1/",
         "data2/data2/",
         "data3/data3/",
         "data4/data4/",
         "data5/data5/",
         "data6/data6/",
		 "data7/data/"
       ]

CSV_FILE = "driving_log.csv"

#data extraction from all source files
DATA=[]
for index in range(0,len(PATH)):
    print(PATH[index]) 
    with open(PATH[index] + CSV_FILE) as csvfile:
        reader = csv.reader(csvfile)
        cntr = 0
        co=0
        for line in reader:
            co+=1
            if (cntr == 0):
                cntr += 1
                continue
            line[0]= PATH[index]+'IMG/'+line[0].split('/')[-1].split('\\')[-1]
            line[1]= PATH[index]+'IMG/'+line[1].split('/')[-1].split('\\')[-1]
            line[2]= PATH[index]+'IMG/'+line[2].split('/')[-1].split('\\')[-1]
            DATA.append(line)
        print(co)

training_data, validation_data = train_test_split(DATA, test_size = 0.15) # splitting data by in 17:3 ratio


total_training_data = len(training_data)
total_valid_data = len(validation_data)
print (total_training_data)
print(total_valid_data)

#flip function
def flip_image(image, measurement): 
    return np.fliplr(image), -measurement

#Image loading and flipping
def get_image(data):

    correction_arr=[0, correction, -correction]
    images=[]
    measurements=[]
        
    for index in range(0,3): 
        try:
            img= cv2.cvtColor(cv2.imread(data[index]), cv2.COLOR_BGR2RGB)
            steering_angle= float(data[3])+ correction_arr[index]

            images.append(img)
            measurements.append(steering_angle)

            fi, fm= flip_image(img, steering_angle)
            images.append(fi)
            measurements.append(fm)
        except:
            print('Image not found')

    return images, measurements

#generator for generating images during training and validation
def generate_samples(data, batch_size):
    shuffle(data)
    while True:
        SIZE = len(data)
        for start in range(0, SIZE, batch_size):
            images, measurements = [], []
            for this_id in range(start, start + batch_size): 
                if this_id < SIZE:
                    image, measurement = get_image(data[this_id])
                    measurements+=measurement
                    images+=image
            yield shuffle(np.array(images), np.array(measurements))



#model architecture
model = Sequential()

#Cropping input images 
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))

#Normalization 
model.add(Lambda(lambda x: (x/127.5)-1)) 

#Layer1 Convolution layer
model.add(Convolution2D(24,5,5, subsample=(2,2)))
model.add(Activation('elu'))

#Layer2 Convolution layer
model.add(Convolution2D(36,5,5,  subsample=(2,2)))
model.add(Activation('elu'))

#Layer3 Convolution layer
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(Activation('elu'))

#Layer4 Convolution layer
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#Layer5 Convolution layer
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#Layer5 Flatten
model.add(Flatten())

#Layer6 Fully connected
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.25))

#Layer7 Fully connected
model.add(Dense(50))
model.add(Activation('elu'))

#Layer8 Fully connected
model.add(Dense(10))
model.add(Activation('elu'))

#Layer9 Fully connected output layer
model.add(Dense(1))
model.summary()

#using Adam optimizer and mean squared error loss
model.compile(loss='mse',optimizer='adam')


#training
print('Training...')
training_generator = generate_samples(training_data, batch_size = BATCH_SIZE)
validation_generator = generate_samples(validation_data, batch_size = BATCH_SIZE)
model.fit_generator(training_generator,samples_per_epoch = total_training_data,validation_data = validation_generator,nb_val_samples = total_valid_data,nb_epoch = NUMBER_OF_EPOCHS,verbose = 1)

#saving model
print('Saving model...')
model.save("model.h5")
print("Model Saved.")

