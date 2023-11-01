import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import time

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')               #Training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')                 #Eval dataset
print(dftrain.head())                               #Prints first 5 values of training dataset
print('\n')
print(dftrain.describe())                           #Statistical analysis of training dataset, including count, mean, standard deviation, minimum values, percentage values, etc. 
print('\n')
print(dftrain.shape)                             #Prints out the shape of training dataset, ie. no of rows/entries followed by no of columns/attributes/features 
print(dfeval.shape)

dftrain.age.hist(bins=20)                           #Creates the histogram of the age graph, with a bin of 20, meaning 20 units on the x axis per column.
plt.show()       

dftrain.sex.value_counts().plot(kind = 'barh')      #Creates a horizontal bar graph showing count of different values of sex, male and female, for vertical, just put kind = 'bar'
plt.show()                                          #Displays all open figures and graphs

dftrain['class'].value_counts().plot(kind='barh')   #Graph of different classes
plt.show()

dftrain.groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')   
#Groups the database by sex, so divides it into male and female, finds the average of the survived row, since its from 0-1, this would be the percentage of survival by sex. 
plt.show()

y_train = dftrain.pop('survived') 
y_eval = dfeval.pop('survived')
#pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#Same as the graph used above, but we merge the 2 datasets since we dropped the values of survived out of the dftrain dataset. 

time.sleep(100000000)

