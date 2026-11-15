#  E-Commerce Demand Forecasting and Inventory Intelligence

##  Project Overview

E-commerce businesses often face challenges in maintaining optimal inventory levels due to fluctuations in customer demand. Overstocking can increase storage costs, while understocking can lead to missed sales opportunities.

This project aims to develop a Machine Learning-based demand forecasting system that predicts future product demand using historical sales data.

The system analyses historical sales patterns, identifies trends, and predicts product demand to support better inventory management and decision-making.

##  Project Objectives

- Analyse historical e-commerce sales data.
- Identify sales trends and demand patterns.
- Perform data cleaning and preprocessing.
- Apply feature engineering techniques to improve prediction.
- Build and evaluate machine learning models.
- Predict future product demand.
- Develop an application for interactive demand prediction.

##  Dataset

**Dataset:** UCI Online Retail Dataset

**Source:** UCI Machine Learning Repository

The dataset contains transactional records from a UK-based online retail store.

Important features include:

- InvoiceDate
- StockCode
- Quantity
- InvoiceNo
- UnitPrice
- CustomerID
- Country

##  Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming language      |
| Pandas       | Data manipulation         |
| NumPy        | Numerical operations      |
| Matplotlib   | Data visualization        |
| Seaborn      | Exploratory Data Analysis |
| Scikit-learn | Machine Learning          |
| Joblib       | Model serialization       |
| VS Code      | Development environment   |

##  Project Workflow

1. **Data Collection:** Obtained the dataset from the UCI Machine Learning Repository.
2. **Data Cleaning:** Handled missing values, duplicates, and inconsistencies.
3. **Exploratory Data Analysis (EDA):** Analysed sales distributions, trends, and outliers.
4. **Feature Engineering:** Created time-based features, lag features, and rolling averages.
5. **Data Splitting:** Used a time-based train-test split to maintain the chronological order of sales data.
6. **Model Building:** Trained different machine learning algorithms.
7. **Model Evaluation:** Evaluated model performance using MAE, MSE, RMSE, and R².
8. **Deployment:** Prepared the trained model for integration into an interactive application.

##  Feature Engineering

The following features were created to capture historical demand patterns:

- Stock Code
- Year
- Month
- Day
- Day of week
- Lag_1
- Lag_7
- Rolling Mean

Lag features help the model understand previous demand, while rolling averages capture historical demand trends.

##  Machine Learning Models

The following algorithms were explored:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The models were evaluated and compared to identify the most suitable approach for demand forecasting.

##  Model Evaluation

Model performance was evaluated using:

| Metric   | Purpose                                                         |
| -------- | --------------------------------------------------------------- |
| MAE      | Measures average absolute prediction error                      |
| MSE      | Measures average squared prediction error                       |
| RMSE     | Measures prediction error in the original unit                  |
| R² Score | Measures how much variation in demand is explained by the model |

##  Deployment

The trained machine learning model is being integrated into an application to provide an interactive interface for demand prediction.

The application aims to allow users to select a product and obtain its predicted demand.

##  Future Improvements

- Improve prediction accuracy through hyperparameter tuning.
- Experiment with additional forecasting algorithms.
- Incorporate more historical sales patterns.
- Develop inventory optimization recommendations.
- Enhance the application interface.
- Deploy the application online.

##  Author

**Arya P. S.**

Economics Graduate | Pursuing PG Diploma in Applied Statistics| Aspiring Data Scientist

Interested in Data Science, Machine Learning, and Business Analytics.
