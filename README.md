# Project 2: Vehicle Sales Price Prediction (Python)

This repository is for the analysis and modeling done on the Kaggle: Vehicle Sales dataset. Below you will find an overview of the data, code, and results. The goal of this project was to create an end-to-end project where I perform an exploratory data analysis, prepare the data (i.e., clean and feature engineer), apply machine learning algorithms to predict the sales price of vehicles, and create a [deployed application with a front end](https://predict-vehicle-price.herokuapp.com/) to productionize the best performing model. The repo for the app can be found [here](https://github.com/MichaelBryantDS/vehicle-price-pred-app).

### Code Used 

**Python Version:** 3.8.11 <br />
**Packages:** pandas, numpy, scipy, sklearn, matplotlib, seaborn, flask, statsmodels, shap, eli5, pickle<br />
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## Vehicle Sales Dataset

The dataset was gathered from [Kaggle](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho). The dataset contains 8 variables and 301 vehicle sales records.

### Variables

`Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`

## Files

### eda-cleaning-engineering.py

This file contains the exploratory data analysis (EDA) and data cleaning. The EDA is performed using descriptive statistics, histograms to determine distributions, a correlation heat map using the Pearson correlation coefficient, and ordinary least squares regression (to determine important variables with p-values and their impact through their coefficients). The cleaning is performed by assigning numbers to strings and features are engineered using dummy variables. The variables are scaled using MinMaxScaler.

### modeling.py

This file contains the modeling where I hyperparameter tune: LinearRegression, Lasso, Ridge, ElasticNet, RandomForestRegressor, GradientBoostingRegressor, SVR, StackingRegressor, VotingRegressor, BaggingRegressor, BaggingRegressor (with pasting), and AdaBoostRegressor. The models are hyperparameter tuned with GridSearchCV based on NMAE and the best models are judged based on MSE, RMSE, MAE, and R-squared metrics. This file also contains code to derive the feature importance from the best models using shap and eli5.

### final-model.py

## Results

### EDA

<div align="center">
  
<figure>
<img src="">
  <figcaption></figcaption>
</figure>
<br/><br/>
  
</div>

### Data Cleaning

I cleaned the data to make the dataset usable for future modeling. I made the following changes:

### Featured Engineering

I feature engineered using the dataset for future modeling. I made the following changes:

## Applications

## Resources

1. [Kaggle: Vehicle dataset](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)
2. [YouTube: Data Science Project from Scratch - Part 5 (Model Building) by Ken Jee](https://www.youtube.com/watch?v=7O4dpR9QMIM)

