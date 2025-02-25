import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("google.csv")

df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], errors='coerce')
df = df.dropna(subset=["date"])
df = df.sort_values("date")

# Selecting the 'Close' price as the target column
scaler = MinMaxScaler(feature_range=(0, 1))
df["close_scaled"] = scaler.fit_transform(df[["close"]])

# Function to create sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

# Splitting the data
df_train = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2020-12-31")]
df_test = df[(df["date"] >= "2021-01-01") & (df["date"] <= "2021-12-31")]

data_train = df_train["close_scaled"].values
data_test = df_test["close_scaled"].values

# Create sequences
X_train, y_train = create_sequences(data_train)
X_test, y_test = create_sequences(data_test)

# Reshape input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Save predictions to CSV
df_test = df_test.iloc[len(df_test) - len(y_test):]
df_test["actual"] = y_test
""" [df_test["predicted"]] = y_pred """
df_test["predicted"] = y_pred.flatten()

df_test.to_csv("testing_predictions_2021.csv", index=False)

# Calculate RMSE, MSE, and R2 Score
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df_test["date"], y_test, label="Actual")
plt.plot(df_test["date"], y_pred, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
