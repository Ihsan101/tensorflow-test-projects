#In this model we use the IMDB movie review dataset from keras, with over 25000 reviews from IMDB already processed and labelled as positive or negative
#Each review is encoded by integers which represents how common a word is in the entire dataset, so a word encoded with 2 will be the second most used word in the entire dataset.


from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np

vocab_size = 88587

maxlen = 250
batch_size = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
print(train_data)

#We need to make all reviews the same length, so we will try to choose a character which is greater than most reviews in the dataset to try and not loose any data. If it is less than 250, it will add 0's to make it 250, if it is more it will delete characters till it is 250. 

train_data = sequence.pad_sequences(train_data, maxlen)
test_data = sequence.pad_sequences(test_data, maxlen)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = 32),
    #Embedding will return a shape with 32 dimensions
    tf.keras.layers.LSTM(32),
    #We need to specify the input dimension of the LSTM layer. 
    tf.keras.layers.Dense(1, activation='sigmoid')
    #Negative and positive, 2 classes, so only 1 output node is required. Sigmoid activation function makes the values between 0 and 1, so we can say all values below 0.5 are negative and above 0.5 are positive
])

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])

history = model.fit(train_data, train_labels, epochs = 10, validation_split = 0.2)
#We will use 20% of the training data to validate it at each epoch. 

result = model.evaluate(test_data, test_labels)
print(result)
#First value is loss, second is accuracy

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], maxlen)[0]

text = input("Enter your review for the movie:")
encoded = encode_text(text)
print(encoded)

reverse_word_index = {value: key for (key,value) in word_index.items()}

def decode_integers(int):
    text = ""
    for num in int:
        if num != 0:
            text += reverse_word_index[num] + " "
    return text[:-1]
print(decode_integers(encoded))

#Make a prediction

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

