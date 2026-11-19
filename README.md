# E-commerce Demand Forecasting and Inventory Intelligence System

## About the Project

E-commerce businesses often face challenges in managing their inventory. Sometimes products run out of stock due to unexpected demand, while at other times businesses maintain more stock than they actually need. Both situations can affect business operations and increase costs.

This project, **E-commerce Demand Forecasting and Inventory Intelligence**, focuses on predicting future product demand using historical sales data and providing inventory recommendations based on those predictions.

The main idea is to use machine learning to understand past sales patterns, estimate future demand and help businesses make better inventory decisions.

## Dataset

For this project, I used the Online Retail Dataset from the UCI Machine Learning Repository. It contains transactional data from a UK-based online retail business.

The dataset includes information such as invoice numbers, product codes, product descriptions, quantities purchased, invoice dates, unit prices and customer details.

I used this data to understand product sales behaviour and identify patterns that could help predict future demand.

## Exploratory Data Analysis

I started the project with Exploratory Data Analysis (EDA) to understand the dataset and identify patterns in the sales data.

I examined the dataset structure, checked for missing values and duplicates, and analysed the distribution of important numerical columns such as Quantity and UnitPrice. I also used different visualizations to understand sales patterns and identify unusual values.

This step helped me understand the quality of the data and decide which preprocessing techniques were required before building the machine learning models.

## Data Preprocessing

After completing EDA, I moved on to data preprocessing to prepare the dataset for modelling.

This involved handling missing values, removing duplicate records, filtering cancelled transactions and dealing with invalid or negative quantities. I also converted the InvoiceDate column into a proper datetime format.

These steps helped prepare the data for further analysis and model development.

## Feature Engineering

Since demand forecasting depends heavily on historical sales behaviour, I created additional features to help the machine learning models understand sales patterns.

I extracted date-related information such as year, month, day and day of the week. I also created lag features to capture previous sales, including demand from the previous day and previous seven, fourteen and twenty-eight days.

Along with these, I calculated rolling averages and rolling standard deviations to understand changes in demand over time.

These features help the model identify patterns in historical sales and use them when predicting future product demand.

## Model Building

After preprocessing and feature engineering, I trained different machine learning regression models to predict product demand.

The models used in this project include Linear Regression, Decision Tree Regressor and Random Forest Regressor.

I also performed hyperparameter tuning using GridSearchCV and RandomizedSearchCV to find suitable model parameters and improve performance.

A baseline model was also used for comparison to understand whether the trained models were actually improving the predictions.

## Model Evaluation

To evaluate the performance of the models, I used different regression metrics, including MAE, MSE, RMSE and R² Score.

These metrics helped me understand the difference between actual and predicted demand and compare the performance of the models.

I compared the evaluation results to identify the strengths and limitations of each model and understand how accurately it could predict product demand.

## Application Development

After completing the modelling process, I developed an interactive application using Streamlit.

The application allows users to select a product using its StockCode, enter the unit price, choose a forecast date and provide information about the current inventory.

Users can also enter historical sales information and specify a safety buffer before generating a demand prediction.

Once the prediction is generated, the application displays the predicted demand, current stock, stock balance and estimated demand for 30 days.

It also provides a visual comparison of predicted demand and available inventory, making the results easier to understand.

## Inventory Intelligence

An important part of this project is connecting demand prediction with inventory management.

Instead of only predicting how many units might be sold, the application also helps estimate how much stock a business should maintain.

The target stock is calculated using predicted demand and the safety buffer.

**Target Stock = Predicted Demand + Safety Buffer**

The application compares the target stock with the current inventory and identifies whether additional stock is required or whether the available inventory is sufficient.

This helps businesses make more informed inventory decisions, avoid potential stock shortages and reduce unnecessary excess stock.

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Jupyter Notebook, Joblib, Git and GitHub.

## Future Improvements

There is still scope to improve this project by exploring more advanced forecasting techniques, incorporating seasonal patterns and improving prediction accuracy.

The application can also be extended with real-time inventory updates and additional features to make it more useful for practical business applications.

## Conclusion

Through this project, I explored how machine learning can be used to solve a practical business problem.

I worked through the complete process, starting from understanding and analysing the dataset to preprocessing, feature engineering, model building, evaluation and application development.

The project helped me understand how historical sales data can be converted into meaningful predictions and how those predictions can be used to support inventory management decisions.

The overall goal is to make demand forecasting more useful by connecting machine learning predictions with real-world business needs.
