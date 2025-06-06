import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Load Data
df = pd.read_csv("covid_19_clean_complete.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar Country Selector
st.sidebar.title("COVID-19 Forecast Settings")
country_list = df['Country/Region'].unique()
selected_country = st.sidebar.selectbox("Select Country", sorted(country_list))

# Filter by country
df_country = df[df['Country/Region'] == selected_country]

# Group by date
df_grouped = df_country.groupby('Date')['Confirmed'].sum().reset_index()

# Create 'Days' feature
df_grouped['Days'] = (df_grouped['Date'] - df_grouped['Date'].min()).dt.days

# Linear Regression Model
X = df_grouped[['Days']]
y = df_grouped['Confirmed']
lr_model = LinearRegression()
lr_model.fit(X, y)
df_grouped['LR_Pred'] = lr_model.predict(X)

# ARIMA Model
arima_model = ARIMA(df_grouped['Confirmed'], order=(5, 2, 1))
arima_result = arima_model.fit()
df_grouped['ARIMA_Pred'] = arima_result.fittedvalues

# Forecast next 30 days
future_days = 30
future_X = np.array(range(df_grouped['Days'].max() + 1, df_grouped['Days'].max() + 1 + future_days)).reshape(-1, 1)
future_lr = lr_model.predict(future_X)
future_dates = pd.date_range(df_grouped['Date'].max() + pd.Timedelta(days=1), periods=future_days)

# ARIMA Forecast
forecast_arima = arima_result.forecast(steps=future_days)

# Combine
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'LR_Forecast': future_lr,
    'ARIMA_Forecast': forecast_arima
})

# Streamlit Interface
st.title("📈 COVID-19 Forecasting")
st.write(f"Forecasting COVID-19 Cases for **{selected_country}**")

# Display current data
st.subheader("Historical Cases")
st.line_chart(df_grouped.set_index('Date')[['Confirmed']])

# Display predictions
st.subheader("Model Predictions (Historical Fit)")
st.line_chart(df_grouped.set_index('Date')[['Confirmed', 'LR_Pred', 'ARIMA_Pred']])

# Display forecast
st.subheader("30-Day Forecast")
st.line_chart(forecast_df.set_index('Date')[['LR_Forecast', 'ARIMA_Forecast']])

# Show raw data option
if st.checkbox("Show raw data"):
    st.write(df_grouped)
