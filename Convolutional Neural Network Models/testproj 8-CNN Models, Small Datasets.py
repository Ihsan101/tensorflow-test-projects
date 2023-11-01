from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#This is a continuation for project 7,Im just making it easier to understand by splitting up data augmentation

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


#What we do in this case scenario, is to zoom in, rotate, and perform other types of transformation on our small existing dataset to turn a small amount of images to a very large amount of data.
datagen = ImageDataGenerator(
    rotation_range=40,          #Rotates at a random angle between -40 and 40 degrees
    #Value of the rest of the arguments will be between 0-1 and float value, they are percent values
    width_shift_range=0.2,      #Width will be shifted randomly between -20% to 20%
    height_shift_range=0.2,     
    shear_range=0.2,            #Uk what shear is from 11th grade physics already, same thing -20% to 20% of shear
    zoom_range=0.2,
    horizontal_flip=True,       #Cant do vertical flip, since some items upside down look like osmething else, so wont really help.
    fill_mode='nearest'
)

test_img = train_images[25]
img = image.img_to_array(test_img)          #Convert image to numpy array
img = img.reshape((1,)+ img.shape)        #First part would be one, then the rest will be decided by shape of the img.

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    #Loop will run forever till we break it, and save the images to current diretory. 
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:            #So 4 images are created
        break

plt.show()