import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the dataset
df = pd.read_csv(r"C:\Users\HP LAPTOP\Desktop\currency\archive (2)\dc.csv")

# View the first few rows and columns of the data
print(df.head())
print(df.columns)

# Adjust column names
df['Price'] = df['close_USD']  # or 'close_SAR' depending on your need

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Unnamed: 0'])

# Set the date as the index
df.set_index('Date', inplace=True)

# Ensure the index is sorted
df = df.sort_index()

# Check for missing values and fill them
df = df.fillna(method='ffill')
print(df.isnull().sum())

# ARIMA model
price_data = df['Price']

# Split the data into train and test sets
train_size = int(len(price_data) * 0.8)
train, test = price_data[:train_size], price_data[train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast with confidence intervals
forecast_results = model_fit.get_forecast(steps=len(test))
forecast = forecast_results.predicted_mean
conf_int = forecast_results.conf_int(alpha=0.05)

# Plot the forecast against actual data
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Price')
plt.plot(test.index, forecast, label='Forecasted Price', color='red')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()

# Scale the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data.values.reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Use a time step of 60 days
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape the data for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10)

# Predict on the test set
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverse scale the predictions

# Plot the LSTM results
plt.figure(figsize=(10, 6))
plt.plot(price_data.index[train_size + time_step + 1:], price_data.values[train_size + time_step + 1:], label='Actual Price')
plt.plot(price_data.index[train_size + time_step + 1:], predictions, label='LSTM Predictions', color='red')
plt.legend()
plt.show()
model.save('lstm_currency_model.h5')