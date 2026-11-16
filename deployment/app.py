
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models
model = joblib.load("demand_model.pkl")
encoder = joblib.load("encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="E-commerce Demand Forecasting",
    
    layout="wide"
)

# Title
st.title("E-commerce Demand Forecasting")
st.write("Demand Prediction and Inventory Intelligence System")

st.divider()

# Product selection
st.subheader("Product Information")

product_code = st.text_input(
    "Enter Product StockCode"
)

col1, col2 = st.columns(2)

with col1:
    forecast_date = st.date_input("Select Forecast Date")

with col2:
    current_stock = st.number_input(
        "Enter Current Stock",
        min_value=0,
        value=100
    )

st.subheader("Historical Sales Information")

col3, col4, col5 = st.columns(3)

with col3:
    lag_1 = st.number_input(
        "Previous Day Sales (Lag 1)",
        min_value=0.0,
        value=10.0
    )

with col4:
    lag_7 = st.number_input(
        "Sales 7 Days Ago (Lag 7)",
        min_value=0.0,
        value=12.0
    )

with col5:
    rolling_mean = st.number_input(
        "7-Day Rolling Average",
        min_value=0.0,
        value=11.0
    )

# Prediction
if st.button("Predict Demand"):

    if not product_code.strip():
        st.warning("Please enter a product StockCode.")

    else:
        try:
            # Encode product
            encoded_product = encoder.transform(
                [[product_code]]
            )[0][0]

            # Prepare input features
            input_data = pd.DataFrame([{
                "StockCode": encoded_product,
                "Year": forecast_date.year,
                "Month": forecast_date.month,
                "Day": forecast_date.day,
                "Dayofweek": forecast_date.weekday(),
                "Lag_1": lag_1,
                "Lag_7": lag_7,
                "Rolling_Mean": rolling_mean
            }])

            # Predict demand
            prediction = model.predict(input_data)[0]

            predicted_demand = max(0, round(prediction))

            # Inventory analysis
            st.divider()
            st.subheader("Forecast Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Demand",
                f"{predicted_demand} units"
            )

            col2.metric(
                "Current Stock",
                f"{current_stock} units"
            )

            stock_difference = current_stock - predicted_demand

            col3.metric(
                "Stock Balance",
                f"{stock_difference} units"
            )

            # Recommendation
            if current_stock < predicted_demand:
                st.error(
                    f"Restocking Required! "
                    f"Consider ordering {predicted_demand - current_stock} units."
                )

            else:
                st.success("Current inventory is sufficient.")

            # Visualization
            chart_data = pd.DataFrame({
                "Category": ["Predicted Demand", "Current Stock"],
                "Quantity": [predicted_demand, current_stock]
            })

            st.subheader("Demand vs Current Stock")

            st.bar_chart(
                chart_data.set_index("Category")
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
