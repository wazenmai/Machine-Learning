#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random



# Global attributes
# Do not change anything here except TODO 1 
StudentID = '107062130' # TODO 1 : Fill your student ID here
input_dataroot = 'input.csv' # Please name your input csv file as 'input.csv'
output_dataroot = StudentID + '_basic_prediction.csv' # Output file will be named as '[StudentID]_basic_prediction.csv'

input_datalist =  [] # Initial datalist, saved as numpy array
output_datalist =  [] # Your prediction, should be 20 * 2 matrix and saved as numpy array
                      # The format of each row should be [Date, TSMC_Price_Prediction] 
                      # e.g. ['2021/10/15', 512]

# You can add your own global attributes here

# Read input csv to datalist
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))

# From TODO 2 to TODO 6, you can declare your own input parameters, local attributes and return parameters
    
def SplitData(data):
    # TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data 
    
    n = data.shape[0]
    test = data[n - 20:, :]
    # test = []

    # train : valid = 8 : 2
    data = data[:n-20, :]
    portion = data.shape[0] // 10
    train = data[0:portion * 8, :]
    valid = data[portion * 8:, :]

    return train, valid, test

def PreprocessData(data):
    # TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    
    # scale the data to small number (avoid overflow)
    # eliminate the date
    date = data[:, 0]
    MTK = data[:, 1].astype(np.float) / 100
    TSMC = data[:, 2].astype(np.float) / 100

    return date, MTK, TSMC

def Regression(X, y, alpha=0.001, epochs=1000):
    # TODO 4: Implement regression
    theta = [0, 0]

    n = float(X.shape[0])
    print("n: ", n)
    last_loss = 100

    for i in range(epochs):
        y_pred = theta[1] * X + theta[0]

        d_theta_1 = sum(X * (y_pred - y)) / n
        d_theta_0 = sum(y_pred - y) / n
        theta[1] = theta[1] - alpha * d_theta_1
        theta[0] = theta[0] - alpha * d_theta_0

        loss = CountLoss(X, y, theta)
        if i % 10 == 0:
            print(f"Epoch {i}: w1 = {theta[1]} w0 = {theta[0]} loss = {loss}, {loss <= last_loss}")
        last_loss = loss

    return theta

def PlotRegressionLine(X, y, theta):
    y_pred = theta[1]*X + theta[0]
    plt.scatter(X, y) 
    plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')  # regression line
    plt.show()


def CountLoss(X, y, theta):
    # TODO 5: Count loss of training and validation data

    y_pred = theta[0] + theta[1] * X
    n = float(X.shape[0])
    loss = 1 / n * sum((y - y_pred) ** 2)
    return loss

def MakePrediction(X, theta):
    # TODO 6: Make prediction of testing data 

    y_pred = theta[0] + theta[1] * X
    y_pred = y_pred * 100
    return y_pred


# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
train_data, valid_data, test_data = SplitData(input_datalist)
print("train: ", train_data.shape)
print("valid: ", valid_data.shape)
print("test: ", test_data.shape)

date_train, X_train, y_train = PreprocessData(train_data)
date_valid, X_valid, y_valid = PreprocessData(valid_data)
date_test, X_test, y_test = PreprocessData(test_data)

theta = Regression(X_train, y_train)
loss = CountLoss(X_train, y_train, theta)
print("train loss: ", loss)
loss = CountLoss(X_valid, y_valid, theta)
print("valid loss: ", loss)
PlotRegressionLine(X_valid, y_valid, theta)

prediction = MakePrediction(X_test, theta)
output_datalist = []

for i in range(len(prediction)):
  output_datalist.append([date_test[i], int(round(prediction[i])])
# Write prediction to output csv
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    for row in output_datalist:
        writer.writerow(row)

