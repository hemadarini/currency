import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\HP LAPTOP\Desktop\currency\archive (2)\dc.csv")

# Adjust column names and set up the DataFrame
df['Price'] = df['close_USD']  # or 'close_SAR' depending on your need
df['Date'] = pd.to_datetime(df['Unnamed: 0'])
df.set_index('Date', inplace=True)
df = df.sort_index()
df = df.fillna(method='ffill')

# Scale the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

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

# Load the saved LSTM model
model = load_model('lstm_currency_model.h5')

# Split the data into training and testing sets


train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

# Predict on the test set
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverse scale the predictions

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size + time_step + 1:], df['Price'].values[train_size + time_step + 1:], label='Actual Price')
plt.plot(df.index[train_size + time_step + 1:], predictions, label='LSTM Predictions', color='red')
plt.legend()
plt.show()
