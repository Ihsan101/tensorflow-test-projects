import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Each of these items are of the dataype numpy array. 

print(train_images.shape)           #Shape will give a result of (6000, 28, 28), so its 6000 images of 28x28 pixel size
#So each image will have 784 pixels.

print(train_images[0,23,23])        #Calling for the first images 23rd column and 23rd row pixel
#Returns the value of 194. Getting each individual pixel like this will return its grayscale value. So this pixel is probably a very light shade of gray, since 0 is black and 255 is white

print(train_labels[:10])
#gives label value of first 10 items, from index 0 - 9. This will be in an integer form of a random number from 0-9. The item which they actually represent are given in the class_names variable

#Now we want to see the image. Cuz uk. We aint braindead like that. Sooooooooo thats why we got matplotlip baby to turn those grayscale values into actual pixels which we can SEEEEEEEEEEEEEEEEEEEEE
plt.figure()
plt.imshow(train_images[1], cmap='gray')       
#By default it will try to make these grayscale images into colored ones, so we gotta specity we want them to be grayscaled
plt.colorbar()                                 
#Just to give us a rough estimate of what each pixels color is by looking at the bar on the right side  
plt.show()

#Time for data processing. We want most input data to be between 0-1 or -1 - 1 as we learnt in NN,
#This is due to the fact by default NN will have weights and bias on a random value from 0-1, so if we have massive input value and tiny weights, itll have to work harder to adjust the weights and biases.
#We know the grayscale values are going to be between 0 - 255, so dividing all of them by 255 will give us the array of numbers consisting of only values between 0 and 1

train_images = train_images / 255.0
test_images = test_images / 255.0

#Time to build model babyyyyy. Well use a keras sequential model. Use tensorflow documentation for more info. Its one of the most basic form of Neural Netowrk models
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),         
    #Input layer
    keras.layers.Dense(128, activation='relu'),
    #Hideen Layer
    keras.layers.Dense(10, activation='softmax')
    #Output layer
])
#First layer takes in the 2D array of size 28x28 and turns it into a 1D array of 784 units, basically flattening down the 2D array into a 1D array

#Dense stands for Dense Neural networks, like I already learnt it just means that each node on this layer is connected to every single node on the previous layer
#Dense Layer 2 is the hidden layer of 128 nodes, and we use an activation function of rectifier linear unit
#3rd layer which is dense is the final output layer, it has the same number of nodes as classes we predicting for, which is 10 nodes. 
#Softmax is an activtion function whcih makes sure all the nodes values add up to 1 and the values are between 0 and 1, to give us a proper probability distribution. 

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
#So uses the adam optimizer along with the loss function given to adjust the weights and bias of the given model, and uses the metric of accuracy to test the overall state of the model after training. 

model.fit(train_images, train_labels, epochs=10)
#Learnt about epochs already, over 10 repetitions of the training data. 

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
#From what i understood, verbose = 0 just means it runs in silent mode, the progress bar wont be displayed in the console. verbose = 1 shows the loss and accuracy along with progress bar in the console already, so the below print statements basically useless, but they arent rounded off values tho. 
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


#Accuracy on the model is lower than accuracy we see during training. So, classic case of overfitting, might have to reduce the number of epochs then. 
#This model is for learning the sequential model, so we wont worry about accuracy too much right now. 




predictions = model.predict(test_images)
#Models are usually better at making predictions on a bulk of images, on top of that the predict only takes array values. Keep that in mind when sending only one image for it to predict. 
print(predictions)
#This will give a list of lists of each item. For the prediction of the first item do predictions[0], it will give the probability distribution of the model, so it will give the percentage chance that it could belong to any of the output classes given. 

print(np.argmax(predictions[0]))
#Returns index of highest value in the list of predictions, so it will give the index number of the class from the prediction of the first object.

print(class_names[np.argmax(predictions[0])])
#Same thing, just getting the actual prediction of the first piece of clothing.




#We can use the image function we used above to evalute our training data to also check if the prediction is right or not. 

#Now we need a verification function, to you know, make this entire thing usable by a common user.

def predict(model, image, correct_label):
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap='gray')     
    plt.title("Expected:"+ label)
    
    if label == guess:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("Guess:"+guess, color = color)
    plt.xlabel("Guess:"+guess)
    plt.colorbar()                                 
    plt.show()

def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return num
        else:
            print("Fam, brother, try again, that aint a valid number.")

while True:
    num = get_number()
    image = test_images[num]
    label = test_labels[num]
    predict(model, image, label)
