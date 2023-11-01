import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import tensorflow as tf
import time

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')               #Training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')                 #Eval dataset

print(type(dftrain))
y_train = dftrain.pop('survived') 
y_eval = dfeval.pop('survived')
#Popped out the value to be predicted, aka the label

CATEGORIAL = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC = ['age', 'fare']
#Subdividing the database columns into categorical or numerical, categorical columns contain string values and need to be converted into numbers to work in regression models. 

feature_columns = []
#Feature columns contain the features, or the inputs used to predict the label

for feature_name in CATEGORIAL:
    vocabulary = dftrain[feature_name].unique() #gets list of all unique values from that column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#Creating an input function, which converts our panda dataframe into a tf.data.Dataset form. 

def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size = 32):
    def input_function():                           #Inner function whose final value shall be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))          #creating tf.data.Dataset object with data and its label value. 
        if shuffle:                                                                 #Checking if we want to shuffle object or not
            ds = ds.shuffle(1000)                                                   #Randomizing order of data, higher the number bigger the randomizing...i think i am not sure
        ds = ds.batch(batch_size).repeat(num_epochs)                                #Split dataset into batches specified and processes for number of epochs. Epcohs is how many times the dataset is repeated. 
        return ds                                                                 
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#CREATING THE MODEL ---------------------------------------------
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
#Creating a linear estimator by passiing the feature columns we created before. 

#TRAINING THE MODEL ---------------------------------------------
linear_est.train(train_input_fn)                    #Training data first 
result = linear_est.evaluate(eval_input_fn)         #Getting result values by evaluating the model

clear_output()                     #Clears outputs given so far in console, just to make everything a bit nicer, since training gives a lot of values in console making it harder to read.
print(result)
print(result['accuracy'])          #The result variable is just a dictionary of stats about our model, so rn we are asking for the accuracy value


#Now trying to precit values, with the same eval function to check the predicted values of each person. 
result = list(linear_est.predict(eval_input_fn))            #This is actually a generator object, meant to be looked through rather than be used as a list, so we had to convert it to a list first. 
print(dfeval.loc[0])                                        #Shows details of the value of the person who we are predicting.                
#Takes the first index item, which is a dictionary of multiple values, takes the probablity value of the same. 0 Depics percentage chance it doesnt survive, 1 is percentage chance it survives. 
print(result[0]['probabilities'][1])                 #Percentage chance it survives.
#To get the result of the next input, just change 0 to 1, or 2, or so on. 
print('')
print(y_eval.loc[0])                                #Checking if he/she actually survived or not

# def input_fn(features, batch_size=256):
#     """An input function for prediction."""
#     # Convert the inputs to a Dataset without labels.
#     return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

# while True:

#     features = ('sex','age','n_siblings_spouses','parch','fare','class','deck','embark_town','alone')
#     print('')
#     sex = input("Enter sex, male or female: \n")
#     age = float(input("Enter age: \n"))
#     n_siblings_spouses = int(input("Enter number of siblings and spouses: \n"))
#     parch = int(input("Enter number of parents and children: \n"))
#     fare = float(input("Enter fare cost: \n"))
#     pclass = input("Enter class, First, Second, or Third: \n")
#     deck = input("Enter deck, unkown or from A - G: \n")
#     embark_town = input("Enter town of embarking, Southampton, Cherbourg or Queenstown: \n")
#     alone = input("Are you alone? y or n: \n")
#     predict = {}
#     list = [sex, age, n_siblings_spouses, parch, fare, pclass, deck, embark_town, alone]
#     for i in features:
#         for j in list:
#             predict[i] = j

#     make_predict_fn = lambda: input_fn(predict)
#     result = linear_est.predict(make_predict_fn)
#     predictions = linear_est.predict(
#     input_fn=lambda: input_fn(predict))

#     for pred_dict in predictions:
#         class_id = pred_dict['class_ids'][0]
#         probability = pred_dict['probabilities'][class_id]
#         print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
#         100 * probability))
     

