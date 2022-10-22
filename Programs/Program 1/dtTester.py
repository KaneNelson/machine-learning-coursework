# Testing the decision tree on the given dataset

from numpy import *
import dtree

# create a decision tree object
dt = dtree.dtree()

# load your data in for building the tree
fileName = input('Data File: ')
data, classData, featureNames = dt.read_data(fileName)
print('Feature Data: ', data)
print('Class Data:', classData)
print('feature Names: ', featureNames)

# build the decision tree model
t = dt.ID3(data,classData,featureNames, "")
print("Tree stored as a dictionary: ", t)

#print out the decision tree model
print("\n-------------------")
print("Decision Tree Model:")
print("-------------------\n")
dt.printTree(t, "")

predicted = dt.classifyAll(t, data)
print("\nPreidction Accuracy: ", dt.predictionAccuracy(predicted, classData))

