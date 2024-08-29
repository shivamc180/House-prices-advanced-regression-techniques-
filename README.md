# House Price Prediction Model

This repository contains a project for predicting house prices based on various features using linear regression. The model is trained on a dataset with multiple features, and polynomial features are used to enhance the model's performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Features Engineering](#features-engineering)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview

The goal of this project is to predict house prices using a linear regression model. The dataset includes various features related to the houses, and the model is trained to predict the house prices based on these features.

## Data Description

The dataset used for this project is `house_data.csv`, which contains 21 columns and 21,613 rows. The columns include:

- **id**: Unique identifier for each house
- **date**: Date of the house listing
- **price**: Target variable - price of the house
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **sqft_living**: Square footage of living space
- **sqft_lot**: Square footage of the lot
- **floors**: Number of floors
- **waterfront**: Binary indicator of whether the house has a waterfront view
- **view**: Quality of the view (0-4 scale)
- **condition**: Condition of the house (1-5 scale)
- **grade**: Overall grade of the house (1-13 scale)
- **sqft_above**: Square footage of the house excluding the basement
- **sqft_basement**: Square footage of the basement
- **yr_built**: Year the house was built
- **yr_renovated**: Year the house was renovated
- **zipcode**: Zip code of the house
- **lat**: Latitude of the house
- **long**: Longitude of the house
- **sqft_living15**: Square footage of living space in 2015
- **sqft_lot15**: Square footage of the lot in 2015

## Features Engineering

1. **Data Preprocessing**:
   - Categorical features are one-hot encoded.
   - Numerical features are scaled using `StandardScaler`.
   - Date features are not included in this project.

2. **Polynomial Features**:
   - Polynomial features of degree 2 are considered, but the final pipeline includes only scaling and one-hot encoding.

## Model Training

1. **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets.
2. **Pipeline Creation**: A pipeline is created to preprocess the data and train the linear regression model.
3. **Model Training**: The linear regression model is trained on the preprocessed training data.

## Evaluation

The model's performance is evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual house prices.
- **R-squared (R²)**: Represents the proportion of the variance in the dependent variable (price) that is predictable from the independent variables.

The model achieved the following performance metrics:
- **Mean Squared Error**: 45,874,072,431.61
- **R-squared**: 0.6966

## Visualization

The performance of the model is visualized with:
- **Predicted vs Actual Plot**: Shows the correlation between predicted and actual house prices.
- **Residuals Plot**: Displays the residuals to check the model's performance.

## Saving and Loading the Model

- **Saving**: The trained model can be saved using `joblib`.
  ```python
  import joblib
  joblib.dump(model, 'Model/linear_regression_model.pkl')
  ```
## Loading: To load the saved model:
```python
import joblib
model = joblib.load('Model/linear_regression_model.pkl')
```
