import pandas as pd
import tensorflow as tf


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
#Downloading required files into cache

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
#Location of the file is the one given above, names of the columns are the one given in the csv file, and the header is the first row. 

train_y = train.pop('Species')
test_y = test.pop('Species')

# The label column has now been removed from the features.

def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Feature columns describe how to use the input.
#This time we dont need to check if it is a categorical or numerical feature column, since all of them are numerical
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#Refer to tensorflow for different type of classifiers, we arent using linear classifiers as this isnt a proper linear model, and we arent using Deep Neural Network Combined Classifiers as the model isnt that wide. So we shall just use a DNN Classifier. 
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

#This given DNN classifier contains 2 hidden layers of 30 and 10 nodes respectively. 

# Training the Model.

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#lambda used to create one line functions, last time in regression models we created a function inside a function so that we could just return the function to be used, this time we use lambda to create a one line function to then be used in the rest of the program. 
#Steps refers to how many times ud go through the dataset till u hit that given number of rows processed
#While training, each step number shall be given along with a number called as loss shall also be defined, try and get this number as low as possible

# Generate predictions from the model
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

print("Please type numeric values as prompted.")
for feature in features:
    val = input(f'{feature}:')
    predict[feature] = [float(val)]

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict))

for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    