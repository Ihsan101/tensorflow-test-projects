import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1, like we did in testproj 6
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print(train_images[0].shape)
#Getting shape of each item in our array, youll see why later. 

#Viewing some of the images one by one, if required
# IMG_INDEX = 0
# plt.imshow(train_images[IMG_INDEX])
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
# plt.show()

#Viewing multiple images at once, 
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#       #Used to hide the x and y axis of the figures
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#We dont need to specify the input shape after the first convalution layer, and the input shape given, (32, 32, 3) is the shape of the items in the CIFAR database.
#We will use 32 filters in this case, of size 3x3 over our input data then use the relu activation function
model.add(layers.MaxPooling2D((2, 2)))
#Downsamples the data into 2x2 samples. Since stride isnt specified, it takes the default value as 2x2, the same as the pooling size. 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#Next layers is just a repeat of the same, but we use 64 filters instead of just 32.

model.summary()
#Getting summary of the model data, this can help us get the shape of outputs of the model in each layer. 


#Now we have extracted features, we shall classify them using dense layers, like a Dense Neural Network. 

model.add(layers.Flatten())
#Done this already, makes the entire array into a 1D array, making it like a list
model.add(layers.Dense(64, activation='relu'))
#64 artifical neuron dense layer with an activation function of relu. 
model.add(layers.Dense(10))
#Final output layer, 10 nodes cuz uk 10 classes. 

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#Compiling just like we would any other type of NN Model
#We using a different loss function this time, until you play around with a few of them, trial and error is the best way to figure out which one works best for you. 

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
#Gives accuracy and loss with training data and validation data, unlike last time where we only gave training data in fitting stage.

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    #Not really required this time, since we already validated the data in fitting stage, so this is just running it once more for no use. 

