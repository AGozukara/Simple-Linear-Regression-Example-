## Linear regression exampLe. 
#Including nessecary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model
#Reading csv file with pandas 
data=pd.read_csv("C:/Users/Ahmet/Desktop/Salary_Data.csv")
# Marking the given points on graph
# Experience which is independent variable is on X-axis and Salary which depends on Experience on Y-axis.
plt.xlabel('Experience(Years)')
plt.ylabel('Salary(TL)')
plt.scatter(data.YearsExperience,data.Salary, color='red', marker='*')
#Creating a linear regression model
reg=linear_model.LinearRegression()
#Fitting the model with given data.
reg.fit(data[['YearsExperience']],data.Salary)
##Plotting the regression model on graph
plt.plot(data.YearsExperience,reg.predict(data[['YearsExperience']]),color='blue')
reg.predict([[1.1]])
