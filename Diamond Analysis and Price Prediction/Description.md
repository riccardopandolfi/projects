# Diamond Price Prediction

This project focuses on analyzing and predicting diamond prices using a dataset called 'diamonds.csv.' The project involves descriptive analysis, identifying the highest quality diamonds, and building a predictive model for diamond prices.

## Dataset
The dataset used for this project is 'diamonds.csv,' which contains information about various attributes of diamonds such as carat, cut, color, clarity, depth, table, price, and more.

## Data Cleaning
To ensure the quality of the data, the following data cleaning steps were performed:
1. Replaced zero values with NA.
2. Removed observations with missing values.

## Descriptive Analysis
Descriptive analysis was performed to gain insights into the dataset. Summary statistics were calculated for the variables in the dataset, and histograms were plotted to visualize the distributions of carat, depth, and price variables. Additionally, frequency tables and bar plots were created to analyze the categorical variables: cut, color, and clarity.

## Identifying Highest Quality Diamonds
To identify the highest quality diamonds, the analysis focused on diamonds with a clarity grade equal to 'IF' (Internally Flawless). Descriptive statistics were calculated for price and carat variables for these diamonds. Furthermore, frequency tables and bar plots were created to analyze the color and cut variables for the highest quality diamonds.

## Correlation Analysis
A correlation analysis was performed on the quantitative variables of the dataset to identify any relationships between them. The correlation matrix was visualized using a correlation plot.

## Predictive Model for Diamond Prices
A predictive model was built to estimate diamond prices based on various attributes. The dataset was preprocessed by creating dummy variables for the categorical variables: cut, color, and clarity. The dataset was split into training and testing sets, and a lasso regression model was trained using cross-validation. The best lambda value was chosen, and the final model was fitted. The model's performance was evaluated using root mean squared error (RMSE) on the training and testing sets. The model's predictions were plotted against the actual prices for both the training and testing sets.

## Summary
This project analyzes a diamond dataset and builds a predictive model for diamond prices. It includes data cleaning, descriptive analysis, identifying the highest quality diamonds, correlation analysis, and the development of a predictive model using lasso regression. The results provide insights into diamond characteristics and offer a model for estimating diamond prices based on their attributes."
