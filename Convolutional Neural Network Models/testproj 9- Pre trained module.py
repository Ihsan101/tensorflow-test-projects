import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()


#Split data manually
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info = True,
    #If it was set to false, the load wouldnt return a dataset information containing versions, features, splits, nums, etc for metadeta
    as_supervised = True
    #Returns it as a 2 tuple structure, as input and label, else would return a dictionary with all values. 
)
print(raw_train)

get_label_name = metadata.features['label'].int2str         #Creates a function object that we can use to get label names

# for image, label in raw_train.take(2):          #Taking the image and label from the first 2 items in raw_train
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))
#     plt.show()

#We gotta make all the images the same size for easy classification, if they are all different it would be a hastle to handle. Better to compress than expand to decrease loss of data. 
img_size = 160
def format_example(image,label):
    image = tf.cast(image, tf.float32)                  #Converting it all into a float value in case its an integer
    image = (image/127.5)-1                             #The values are then divided by half of 255, so it will give a value between 0 and 2, and then 1 is subtracted so it gives a value between -1 and 1
    image = tf.image.resize(image,(img_size, img_size))
    return image, label

train = raw_train.map(format_example)                   #map takes every single example of raw_train and applies the function given in brackets to it
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# for image, label in train.take(2):
#     plt.figure()
#     plt.imshow(image)
#     plt.show()
        #Images will be a bit messed up since our values are between -1 and 1 and not 0 and 1, and we didnt directly divide with 255. 

img_shape = (img_size, img_size, 3)
#3 stands for the 3 different depth layers due to RGB values. 

base_model = tf.keras.applications.MobileNetV2(input_shape = img_shape,
                                               include_top=False,
                                               #Should we include the classifier with this pretained model or not, in this case we shall be fine tuning it with our own classifier
                                               weights = 'imagenet')
#MobileNetV2 is a pretrained model, MobileNetV3 has been released, but I am not sure how it works properly yet, and if I should use Small or Large version. Play around with it, then start to use it. 
#This model has been trained on 1.4 million images on thousands of different objcets, but we just need cat and dog classifier, so we wont use it. 

#Now our original (1, 160, 160, 3) will output a feature extraction of (32, 5, 5 , 1280) shape from the base_model. 32 means it will have 32 different filters/features
base_model.summary()

batch_size = 32
shuffle_buffer_size = 1000
train_batches = train.shuffle(shuffle_buffer_size).batch(batch_size)
validation_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)

#Freezing the base of our neural network, means to disable the training property of a layer. We need to do this so we dont change the value of the weights and bias of any layers.
base_model.trainable = False
#we have now frozen our model.

#Now we add the classifier. Instead of using Flattening we shall use a tool to make the 5x5 area of each 2D feature map into a single 1280 element vector per filter
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)
#We just need one node since we just predicting between dogs and cats

model = tf.keras.Sequential([
    base_model, 
    global_average_layer, 
    prediction_layer
])

base_learning_rate = 0.0001                 #Defines how much the weights and biases of this new model can be modified. We set the value to be very low
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr = base_learning_rate),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),      #We using Binary Cross Entropy cuz 2 classes, for more than 2 we use CrossEntropy          
    metrics = ['accuracy']
)


initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches,
                    epochs = initial_epochs,
                    validation_data = validation_batches)

acc = history.history['accuracy']
print(acc)

# model.save("dogs_vs_cats.h5")
#Save the model for future use, .h5 is the format used by keras
# new_model = tf.keras.models.load.model('dogs_vs_cats.h5')
#Load a model you have previously saved on ur pc