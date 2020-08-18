# Import Pandas
import pandas as pd
# importing libraries for the plots.
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_name = "iris.csv"
name = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
iris = pd.read_csv(file_name, sep=",", names=name, header=0)
iris = iris.sample(frac=1).reset_index(drop=True)

# I have used first 70 data as the train data.
train_data = iris.loc[0:69].Species
# I have used last 30 data as the test data.
test_data = iris.loc[70:].Species

# Their values(setosa or virginica) is stored in the variable temp.
temp = train_data.values

# temp variable is transformed to an array using np.array
temp = np.array(temp)

# I am finding the indexes where setosa occurred. (This is done to have a better visualization in plots by
# distinguishing the classes of data)
setosa_indexes = np.where(temp == 'setosa')

# The same is done for the other class
virginica_indexes = np.where(temp == 'virginica')

# creating a scatter plot of "Species" using "SepalLengthCm" and "SepalWidthCm" features
plt.title('scatter plot of "Species" using "SepalLengthCm" and "SepalWidthCm"', color='b')
plt.plot(iris.loc[setosa_indexes].SepalLengthCm, iris.loc[setosa_indexes].SepalWidthCm, '.', color='r')
plt.plot(iris.loc[virginica_indexes].SepalLengthCm, iris.loc[virginica_indexes].SepalWidthCm, '.', color='b')
plt.legend(['setosa', 'virginica'])
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.show()
# creating a scatter plot of "Species" using "SepalLengthCm" and "PetalLengthCm" features
plt.title('scatter plot of "Species" using "SepalLengthCm" and "PetalLengthCm"', color='b')
plt.plot(iris.loc[setosa_indexes].SepalLengthCm, iris.loc[setosa_indexes].PetalLengthCm, '.', color='r')
plt.plot(iris.loc[virginica_indexes].SepalLengthCm, iris.loc[virginica_indexes].PetalLengthCm, '.', color='b')
plt.legend(['setosa', 'virginica'])
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()
# creating a scatter plot of "Species" using "SepalLengthCm" and "PetalWidthCm" features
plt.title('scatter plot of "Species" using "SepalLengthCm" and "PetalWidthCm"', color='b')
plt.plot(iris.loc[setosa_indexes].SepalLengthCm, iris.loc[setosa_indexes].PetalWidthCm, '.', color='r')
plt.plot(iris.loc[virginica_indexes].SepalLengthCm, iris.loc[virginica_indexes].PetalWidthCm, '.', color='b')
plt.legend(['setosa', 'virginica'])
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalWidthCm')
plt.show()
# creating a scatter plot of "Species" using "SepalWidthCm" and "PetalLengthCm" features
plt.title('scatter plot of "Species" using "SepalWidthCm" and "PetalLengthCm"', color='b')
plt.plot(iris.loc[setosa_indexes].SepalWidthCm, iris.loc[setosa_indexes].PetalLengthCm, '.', color='r')
plt.plot(iris.loc[virginica_indexes].SepalWidthCm, iris.loc[virginica_indexes].PetalLengthCm, '.', color='b')
plt.legend(['setosa', 'virginica'])
plt.xlabel('SepalWidthCm')
plt.ylabel('PetalLengthCm')
plt.show()
# creating a scatter plot of "Species" using "SepalWidthCm" and "PetalWidthCm" features
plt.title('scatter plot of "Species" using "SepalWidthCm" and "PetalWidthCm"', color='b')
plt.plot(iris.loc[setosa_indexes].SepalWidthCm, iris.loc[setosa_indexes].PetalWidthCm, '.', color='r')
plt.plot(iris.loc[virginica_indexes].SepalWidthCm, iris.loc[virginica_indexes].PetalWidthCm, '.', color='b')
plt.legend(['setosa', 'virginica'])
plt.xlabel('SepalWidthCm')
plt.ylabel('PetalWidthCm')
plt.show()
# creating a scatter plot of "Species" using "PetalLengthCm" and "PetalWidthCm" features
plt.title('scatter plot of "Species" using "PetalLengthCm" and "PetalWidthCm"', color='b')
plt.plot(iris.loc[setosa_indexes].PetalLengthCm, iris.loc[setosa_indexes].PetalWidthCm, '.', color='r')
plt.plot(iris.loc[virginica_indexes].PetalLengthCm, iris.loc[virginica_indexes].PetalWidthCm, '.', color='b')
plt.legend(['setosa', 'virginica'])
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.show()
