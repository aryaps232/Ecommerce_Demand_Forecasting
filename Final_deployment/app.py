import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Demand Forecasting",
    layout="wide"
)

st.title("E-commerce Demand Forecasting")
st.caption("Demand Prediction and Inventory Intelligence System")
st.divider()

MODEL_PATH = "demand_model.pkl"
HISTORY_PATH = "recent_history.csv"

# ---------------------------------------------------------
# Load the model and recent product history
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find {MODEL_PATH}. Keep it in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_history():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(
            f"Cannot find {HISTORY_PATH}. Keep it in the same folder as app.py."
        )
    data = pd.read_csv(HISTORY_PATH, parse_dates=["InvoiceDate"])
    data["StockCode"] = data["StockCode"].astype(str)
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"]).dt.normalize()
    return data.sort_values(["StockCode", "InvoiceDate"])

try:
    artifact = load_model()
    history = load_history()
except Exception as error:
    st.error(str(error))
    st.info(
        "Place app.py, demand_model.pkl, and recent_history.csv in the same folder. "
        "Run the final export cell in your modelling notebook first."
    )
    st.stop()

model = artifact["model"]
encoder = artifact["encoder"]
feature_order = artifact["features"]

# The training notebook encoded StockCode with an OrdinalEncoder.
known_products = sorted(history["StockCode"].dropna().astype(str).unique())

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.header("Product Information")

with st.form("forecast_form"):
    stock_code = st.selectbox(
        "Select Product StockCode",
        options=known_products,
        help="Choose a product that exists in recent_history.csv."
    )

    price_col, date_col = st.columns(2)

    with price_col:
        unit_price = st.number_input(
            "Enter Unit Price (₹)",
            min_value=0.0,
            value=25.00,
            step=0.50,
            format="%.2f"
        )

    with date_col:
        forecast_date = st.date_input(
            "Select Forecast Date",
            value=date.today(),
            format="DD/MM/YYYY",
            help="Choose the date for which you want to estimate demand."
        )

    stock_col, buffer_col = st.columns(2)

    with stock_col:
        current_stock = st.number_input(
            "Enter Current Stock",
            min_value=0,
            value=98,
            step=1
        )

    with buffer_col:
        safety_buffer = st.number_input(
            "Safety Buffer Stock (units)",
            min_value=0,
            value=5,
            step=1
        )

    st.header("Historical Sales Information")
    lag_col1, lag_col2, rolling_col = st.columns(3)

    # Suggest defaults from the most recent history before the selected date.
    product_history = history[history["StockCode"] == str(stock_code)].copy()
    before_date = product_history[
        product_history["InvoiceDate"] < pd.Timestamp(forecast_date)
    ].sort_values("InvoiceDate")

    if len(before_date) >= 30:
        default_lag1 = float(before_date.iloc[-1]["Quantity"])
        default_lag7 = float(before_date.iloc[-7]["Quantity"])
        default_rolling7 = float(before_date.tail(7)["Quantity"].mean())
    else:
        default_lag1, default_lag7, default_rolling7 = 0.0, 0.0, 0.0

    with lag_col1:
        lag_1_input = st.number_input(
            "Previous Day Sales (Lag 1)",
            min_value=0.0,
            value=default_lag1,
            step=1.0
        )

    with lag_col2:
        lag_7_input = st.number_input(
            "Sales 7 Days Ago (Lag 7)",
            min_value=0.0,
            value=default_lag7,
            step=1.0
        )

    with rolling_col:
        rolling_7_input = st.number_input(
            "7-Day Rolling Average",
            min_value=0.0,
            value=default_rolling7,
            step=0.5
        )

    submitted = st.form_submit_button(
        "Predict Demand",
        use_container_width=False
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    # We need 30 prior daily observations because the trained model
    # also uses Lag_14, Lag_28, Rolling_Mean_14, Rolling_Mean_30,
    # Rolling_Std_7 and Sale_Days_7.
    before_date = (
        history[
            (history["StockCode"] == str(stock_code))
            & (history["InvoiceDate"] < pd.Timestamp(forecast_date))
        ]
        .sort_values("InvoiceDate")
    )

    if len(before_date) < 30:
        st.error(
            "There are fewer than 30 historical daily records for this product "
            "before the selected forecast date. Choose a later date or use a "
            "product with at least 30 prior days of history."
        )
        st.stop()

    quantities = before_date["Quantity"].astype(float).to_numpy()

    # Build every feature used during model training.
    # Lag 1, Lag 7 and 7-day rolling average use the values entered by the user.
    forecast_timestamp = pd.Timestamp(forecast_date)
    iso_week = int(forecast_timestamp.isocalendar().week)

    feature_values = {
        "StockCode": str(stock_code),
        "UnitPrice": float(unit_price),
        "Lag_1": float(lag_1_input),
        "Lag_7": float(lag_7_input),
        "Rolling_Mean": float(rolling_7_input),
        "Lag_14": float(quantities[-14]),
        "Lag_28": float(quantities[-28]),
        "Rolling_Mean_14": float(np.mean(quantities[-14:])),
        "Rolling_Mean_30": float(np.mean(quantities[-30:])),
        "Rolling_Std_7": float(np.std(quantities[-7:], ddof=1)),
        "Sale_Days_7": float(np.sum(quantities[-7:] > 0)),
        "Month": int(forecast_timestamp.month),
        "Day": int(forecast_timestamp.day),
        "DayOfWeek": int(forecast_timestamp.dayofweek),
        "WeekOfYear": iso_week,
        "IsWeekend": int(forecast_timestamp.dayofweek >= 5),
    }

    input_df = pd.DataFrame([feature_values])

    # Encode StockCode with the exact encoder saved by the notebook.
    input_df[["StockCode"]] = encoder.transform(input_df[["StockCode"]])

    # Preserve the exact feature order used while training.
    input_df = input_df[feature_order]

    prediction = float(model.predict(input_df)[0])
    predicted_demand = max(0, int(math.ceil(prediction)))

    target_stock = predicted_demand + int(safety_buffer)
    stock_balance = int(current_stock) - predicted_demand
    additional_stock = max(0, target_stock - int(current_stock))
    thirty_day_estimate = predicted_demand * 30

    st.divider()
    st.header("Prediction Results")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Predicted Demand", f"{predicted_demand} units")
    metric2.metric("Current Stock", f"{int(current_stock)} units")
    metric3.metric("Stock Balance", f"{stock_balance} units")

    if current_stock >= target_stock:
        st.success("Current inventory is sufficient, including the safety buffer.")
    elif current_stock >= predicted_demand:
        st.warning(
            f"Demand can be met, but the safety buffer is not covered. "
            f"Consider adding {additional_stock} units."
        )
    else:
        st.error(
            f"Potential stock shortage. You need approximately "
            f"{additional_stock} additional units to reach target stock."
        )

    result1, result2 = st.columns(2)
    result1.metric("Estimated Demand for 30 Days", f"{thirty_day_estimate} units")
    result2.metric("Target Stock (Demand + Buffer)", f"{target_stock} units")
    st.metric("Additional Stock to Reach Target", f"{additional_stock} units")

    # Simple comparison chart
    chart_data = pd.DataFrame({
        "Inventory Measure": ["Current Stock", "Predicted Demand", "Target Stock"],
        "Units": [int(current_stock), predicted_demand, target_stock]
    })

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_data["Inventory Measure"],
                y=chart_data["Units"],
                text=chart_data["Units"],
                textposition="outside"
            )
        ]
    )
    fig.update_layout(
        title="Demand vs Current Stock and Target Stock",
        template="plotly_dark",
        yaxis_title="Units",
        xaxis_title="",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View model input values"):
        st.dataframe(input_df, use_container_width=True)

st.caption(
    "Note: The 30-day demand figure is a simple 30 × one-day prediction estimate, "
    "not a separate 30-day forecast. The model was trained using historical daily "
    "sales features from your notebook."
)
