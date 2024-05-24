import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('Advertising.csv')
X=df.drop('Sales',axis=1)
y=df['Sales']
X_train ,X_test, Y_train, Y_test= train_test_split(X,y,test_size=0.33 , random_state=42)
model=LinearRegression()
model.fit(X_train, Y_train)
# make prediction on the test set
Y_pred = model.predict(X_test)
mae= mean_absolute_error(Y_test,Y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, Y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")
# Print the mean of the Sales column
print(f"Mean of Sales: {df['Sales'].mean()}")
