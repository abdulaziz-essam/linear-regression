import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import os

# Check if the file exists
file_path = 'Advertising.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} does not exist.")

# Load the dataset
df = pd.read_csv(file_path)

# Check for empty dataset
if df.empty:
    raise ValueError("The dataset is empty.")

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Prepare the data
X = df.drop('Sales', axis=1)
y = df['Sales']

# Print columns of X to understand the expected features
print("Columns of X:", X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the mean of the Sales column
print(f"Mean of Sales: {df['Sales'].mean()}")

# Deploy the model
final_model = LinearRegression()
final_model.fit(X, y)

# Coefficients interpretation
print("Model Coefficients:", final_model.coef_)

# Save the model
dump(final_model, 'final_sales_model.joblib')

# Load the model
loaded_model = load('final_sales_model.joblib')

# Predict sales for a new campaign
campaign = pd.DataFrame([[149, 22, 12]], columns=['TV', 'Radio', 'Newspaper'])

# Ensure columns match what the model was trained on
print("Columns of campaign DataFrame:", campaign.columns)

# Predicting sales
new_campaign_sales = loaded_model.predict(campaign)
print("Predicted sales for the new campaign:", new_campaign_sales)
print(df.head())
