# Import Pandas

# This library is imported to use data_frame + reading data
import pandas as pd

# This class is imported as my second arbitrary method of classifying
from perceptron import Perceptron

# Numpy is the most obvious library for various applications! (e.g. sqrt, array, where & etc.)
import numpy as np

# As it is required to calculate the euclidean distance in 4-D space between data points, we take advantage of defining
# a function to do so when it is called with its arguments being data points of interest.


def euclidean_distance(point_1, point_2):
    distance = np.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2 +
                       (point_2[2] - point_1[2]) ** 2 + (point_2[3] - point_1[3]) ** 2)
    return distance


# This piece of code is copied from jupyter notebook to load data properly.

# Load data
file_name = "iris.csv"
name = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
iris = pd.read_csv(file_name, sep=",", names=name, header=0)
iris = iris.sample(frac=1).reset_index(drop=True)

# I have used first 70 data as the train data.
train_data = iris.loc[0:69]
# I have used last 30 data as the test data.
test_data = iris.loc[70:100]

# Defining an array to store the predicted species for test data.
test_result = []

# This is the number of closest points of train data that are chosen to our test data. It is set to 5 as suggested
# by the problem itself
k = 5

# These two counters are defined so as to be increased when one of k closest points of train data is setosa or virginica
# and before iterating the next loop, they are compared and will bring us the predicted result for test data.
setosa_counter = 0
virginica_counter = 0

for i in range(70, 100):
    # for each test data this array is its distance to all train data.
    distances_from_train_data = []

    # These counters which were described a few lines before, need to be reset to 0 when a loop starts.(A new test data)
    setosa_counter = 0
    virginica_counter = 0

    # test data features are stored in temp_point1 to be passed to distance calculator function later.
    temp_point1 = [test_data.SepalLengthCm[i], test_data.SepalWidthCm[i], test_data.PetalLengthCm[i],
                   test_data.PetalWidthCm[i]]

    for j in range(0, 70):

        # train data features are stored in temp_point2 to be passed to distance calculator function later.
        temp_point2 = [train_data.SepalLengthCm[j], train_data.SepalWidthCm[j], train_data.PetalLengthCm[j],
                       train_data.PetalWidthCm[j]]

        # using append attribute of arrays, distance of test data to train data is calculated and stored in the array
        #  using our defined function at the top of the code
        distances_from_train_data.append(euclidean_distance(temp_point1, temp_point2))

    # unsorted distances are stored in a temp variable to keep our access to species of closest point of train data.
    temp = np.array(distances_from_train_data)
    # distances are sorted to pick their first k elements and extract their labels
    distances_from_train_data.sort()

    # In this for loop we decide whether the test data is 'setosa' or 'virginica'
    for m in range(0, k):
        # Index of the first k close points to test data in unsorted array are being derived.
        index = np.min(np.where(temp == distances_from_train_data[m]))

        # Using index we can count how many of these close points belong to each class.
        if train_data.Species.values[index] == 'setosa':
            setosa_counter += 1
        if train_data.Species.values[index] == 'virginica':
            virginica_counter += 1

    # Comparing the counters and decision of this test data's result.
    if setosa_counter > virginica_counter:
        test_result.append('setosa')
    else:
        test_result.append('virginica')

# This counter is increased whenever a test datum's predicted species agrees with our actual labels. Otherwise it won't
# change. The ratio of this counter to the number of test data is our accuracy and precision of prediction or learning.
score_counter = 0
for i in range(len(test_data.Species)):
    if test_result[i] == test_data.Species[i + 70]:
        score_counter += 1

score = score_counter/len(test_data.Species)
print('Score using the first method (Euclidean Distance) is:')
print(score * 100)

# ####################################### Classifier second method #######################################
labels = []
training_inputs = []

# training data is prepared to be used in perceptron.train according to Perceptron object defined in perceptron.
for i in range(len(train_data.Species)):
    training_inputs.append(np.array([train_data.SepalLengthCm[i], train_data.SepalWidthCm[i],
                                     train_data.PetalLengthCm[i], train_data.PetalWidthCm[i]]))

# labels are prepared to be used in perceptron.train according to Perceptron object defined in perceptron.
for i in range(len(train_data.Species)):
    if train_data.Species[i] == 'setosa':
        labels.append(1)
    else:
        labels.append(0)

# Our perceptron has 4 distinct features and its other parameters are set to default value.
perceptron = Perceptron(4)

# Perceptron is trained using transformed to binary labels and merged features.
perceptron.train(training_inputs, np.array(labels))

# variable inputs, contains test data and is prepared to be used in perceptron.predict according to Perceptron object
# defined in perceptron.
inputs = []
for i in range(70, 100, 1):
    inputs.append(np.array([test_data.SepalLengthCm[i], test_data.SepalWidthCm[i],
                            test_data.PetalLengthCm[i], test_data.PetalWidthCm[i]]))
# variable results, contains test data predicted binary labels
result = []
for i in range(0, 30, 1):
    result.append(perceptron.predict(inputs[i]))

# From here on, we calculate the precision using the same method of previous section.
# Prediction is correct under two conditions which are explained in the report. (They're simple though)
second_method_counter = 0
for i in range(0, 30, 1):
    if (result[i] == 1 and test_data.Species[i + 70] == 'setosa') or\
            (result[i] == 0 and test_data.Species[i + 70] == 'virginica'):
        second_method_counter += 1

score_method_2 = second_method_counter/len(test_data.Species)
print('Score using the second method (Linear Perceptron) is:')
print(score_method_2 * 100)
