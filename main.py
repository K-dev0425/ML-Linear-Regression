import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read .csv into a DataFrame
house_data = pd.read_csv("house_prices.csv")
size = house_data['sqft_living']
price = house_data['price']

# machine learning handles arrays not data-frames
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# training
model = LinearRegression()
model.fit(x, y)

# MSE & R value
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x, y))

# get b values
# b0
print(model.coef_[0])
# b1 in our model
print(model.intercept_[0])

# Visualize
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='blue')
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

# Predicting the prices
print("Prediction by the model: ", model.predict([[5000]]))