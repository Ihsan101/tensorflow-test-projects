from keras.preprocessing import sequence
import keras
import tensorflow as tf
import numpy as np
import os

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read in binary mode, then decode it for utf-8 format
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#text is in string form, carrying the given play written by Shakespeare
print(f'Length of text: {len(text)} characters')


vocab = sorted(set(text))
#set(text) will give all alphabets, characters, and formatting used in the text variable, order would be random, so we need to sort it.
# print(set(text))
# print(sorted(set(text)))

char2idx = {u:i for i, u in enumerate(vocab)}
#Enumerate returns a pair of count and the index of the object, so we can use this to get a dictionary where the object is the key and the index is the value
idx2char = np.array(vocab)
#In this case we just take the object in an array. The position of its index will be its value so we dont need to worry about it. 

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
print(text_as_int)

# Just to test out the encoding of the first 25 letters
# print("Text:", text[:25])
# print("Encoded", text_to_int(text[:25]))

def int_to_text(ints):
    try:
        ints = ints.numpy()

    except:
        pass
    return ''.join(idx2char[ints])
#If the item isnt already in numpy array, it will try to convert it into one. Then return the entire thing, joining everyhing with '', aka nothing. Get the character value from the idx2char item

seq_length = 100                                            #Length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

#Creating training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)  #Use map to apply function to every single entry

batch_size = 64
vocab_size = len(vocab)
embedding_dim = 256                 #How big we want every single vector to represent our words as in the embedding layer
rnn_units  = 1024                   #
buffer_size = 1000

#Buffer size to shuffle datasets. TF Data works in infinite sequences, so it doesnt just shuffle sequence and store it in memory, instead it keeps a buffer where the elements are shuffled

data = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder = True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape = [batch_size, None]),            #We dont know how long each batch will be, during training we know the length is 100, but during the prediction phases we dont so we wont add that value here
        tf.keras.layers.LSTM(rnn_units,                                             #No. of nodes = rnn_units. Thats a lotta nodes.
                             return_sequences=True,                                 #We want output at every single time step instead of just the very last one
                             stateful= True,
                             recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)                                           #We want final layer, aka output layer to have same amount of nodes as the amount of total characers whcih it can output, so each node provides a probability distribution. 
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
# model.summary()
#Final output shape = 64, 100, 65      batch size, sequence length, vocab size
#So it will be an array, with 100 lists with 65 elements each, repeated 64 times for each batch. The 100 is for each time stap, 65 for prediction of each character. 

#There is no built in loss function that can check 3D Nested array, so we need to create our own.  

for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)              #Getting prediction of first batch of training data
    #size - 64, 100, 65
    
pred = example_batch_predictions[0]
#size - 100, 65

time_pred = pred[0]
#size - 65

#Now we need to sample the values of the prediction at each timestamp. We cant take the one with the highest probability to create the loss function, since we might risk getting the model stuck in a loop
#Instead, we use sample the categorical distribution we have right now to try and get a value depending on the probabilitie distribution
sampled_indices = tf.random.categorical(pred, num_samples = 1)
# sampled_indices = np.reshape(sampled_indices, (1,-1))[0]
print(np.reshape(sampled_indices, (1,-1))[0])               #1, -1 means to use whatever index it would take to flatten it. It would still be in a indented array, so to only get a singular array we use [0]

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    #from_logits specifies that the predicted values in this case is a logits tensor, which is the vector of raw predictions, since we havent trained the model yet. The values of logits could be less than 0 or greater than 1, unlike normal probabilities
    
model.compile(optimizer='adam', loss = loss)

#Now we are going to setup checkpoints as it trains so that we can load our model from these different checkpoints to continue training it

#Directoy of location of checkpoints
checkpoint_dir = './training_checkpoints'
#Name of checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True
)

history = model.fit(data, epochs = 40, callbacks = [checkpoint_callback])

#Rebuilding the model with a new batch size of 1, instead of a batch size of 64. So we dont need to provide the model with 64 inputs, we just need to provide it with 1
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

#Now we find the latest checkpoint to give us the best weights possible for our model
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape(1, None))

#The following code is just to load a checkpoint which you think would be the best possible fit for the same. Its to load a custom checkpoint
# checkpoint_num = input("Enter checkpoint number to use:")
# model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
# model.build(tf.TensorShape(1, None))

#Now we need to generate the input. 
def generate_text(model, start_string):

    #Number of characters to generate:
    num_generate = int(input("Enter number of characters to generate:\n"))
    
    #Converting our start string to vectors, aka pre processing data before it can be used
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)              #Turn the list of all numbers into a nested list as thats what the model expects to be an input

    #Empty string to store our string
    text_generated = []

    #Setting temperature of the model. 
    temperature = float(input("Enter temperature of the model:\nLower temperature will give more predictable text, higher temp will give more suprising text: \n"))

    model.reset_states()
    #The model will remember the last state used while training the model, which will become problematic since we changed the batch size to 1 now
    for i in range(num_generate):
        predictions = model(input_eval)
        #Now we remove the extra dimension to make it a singular array instead of a nested one
        predictions = tf.squeeze(predictions, 0)

        #Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        #pass the predicted character as the next input to the model along with previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

#Finally getting an output babyy
inp = input("Type starting string for the play to generate: \n")
print(generate_text(model, inp))