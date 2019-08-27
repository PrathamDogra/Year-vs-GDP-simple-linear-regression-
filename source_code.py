""" It is a simple linear regression model which can predict the GDP of India"""
#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Year vs GDP - Sheet1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Train set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

#fitting our simple linear regressor model to our train set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting GDP from our test set of years

Y_pred =regressor.predict(X_test)

#plotting the train set

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train) , color='black')
plt.title("YEAR VS GDP(TRAIN SET)")
plt.xlabel("Year")
plt.ylabel("GDP in billion $")
plt.show()

#plotting the test set

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train) , color='black')
plt.title("YEAR VS GDP(TEST SET)")
plt.xlabel("Year")
plt.ylabel("GDP in billion $")
plt.show()