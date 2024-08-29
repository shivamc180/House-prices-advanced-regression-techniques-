# PRODIGY_ML_01

I have used the Google colab for performing this Task 1 and i have used the dataset like : 
https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv

GG Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

**Step 1:** 

!pip install pandas==1.5.3

**Step 2** 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

**Step 3**

# Load the dataset
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(url)

# Display the first few rows of the dataset
df.head()

**Step 4 :**

# For demonstration purposes, let's create a sample dataframe with relevant features
# You should replace this with the actual columns from your dataset
df = df[['rm', 'age', 'tax', 'medv']]
df.columns = ['square_footage', 'bedrooms', 'bathrooms', 'price']

# Check for missing values
df.isnull().sum()

# Drop rows with missing values (if any)
df.dropna(inplace=True)

Step 5 : 

# Split the data into features and target variable
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 6 :

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

Step 7 : 

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse}')

# Plot the actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
