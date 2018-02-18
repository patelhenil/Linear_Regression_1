import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
#Here we are getting data from out text file and setting it up as 2D datatfram of rows and columns
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#our goal is given brain weight of the new animal we are gonna be able to output animals body weight.
#Our friend here is linear regression. With that we could get the best fit line and then we could predict our goal.

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results Here
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
