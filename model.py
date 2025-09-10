import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
import time
#print("hello i am from model.py")
# Define paths for saving/loading the model and scaler
# The .keras extension is the recommended format for TensorFlow models
MODEL_SAVE_PATH = 'lstmTwo.keras'
SCALER_SAVE_PATH = 'scalerTwo.pkl'

# --- Fixed Hyperparameters (No longer user-editable) ---
# These values are set to provide a reasonable starting point for the model.
FIXED_LOOK_BACK = 125 # Number of previous time steps to use as input features to predict the next time step.
FIXED_EPOCHS = 150    # Number of times the learning algorithm will work through the entire training dataset.
FIXED_BATCH_SIZE = 32 # Number of samples per gradient update.
FIXED_VALIDATION = 0.05 # Percentage of validation from model.fit

class StockModel:
    """
    Handles all the machine learning logic for the stock prediction application.
    This includes training the LSTM model, making predictions, and managing model files.
    """
    def __init__(self):
        """Initializes the model, scaler, and file paths."""
        self.model = None
        self.scaler = None
        
    def check_model_exists(self):
        """Checks if the model and scaler files exist."""
        return os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH)

    def train_model(self, train_df):
        """
        Trains the LSTM model on the provided training dataframe.

        Args:
            train_df (pd.DataFrame): The dataframe containing the training data.

        Returns:
            tuple: A tuple containing the training history object and the training duration in seconds.
        """
        # Use fixed hyperparameters
        LOOK_BACK = FIXED_LOOK_BACK
        EPOCHS = FIXED_EPOCHS
        BATCH_SIZE = FIXED_BATCH_SIZE

        # Use only 'close' price for training
        train_close = train_df['Close'].values.reshape(-1, 1)

        # Initialize and fit the MinMaxScaler on the training data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array_scaled = self.scaler.fit_transform(train_close)

        # Prepare Training Data: Create sequences (X) and corresponding labels (y)
        x_train, y_train = [], []
        for i in range(LOOK_BACK, len(data_training_array_scaled)):
            x_train.append(data_training_array_scaled[i - LOOK_BACK:i])
            y_train.append(data_training_array_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape x_train for LSTM input: (samples, time_steps, features)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM Model
        self.model = Sequential([
            Input(shape=(x_train.shape[1], 1)),
            LSTM(units=100, return_sequences=True),
            Dropout(0.3),
            LSTM(units=100, return_sequences=False),
            Dropout(0.3),
            Dense(units=1)
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])

        # Early Stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        
        # --- Time tracking starts ---
        start_time = time.time()
        
        # Train Model
        history = self.model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            validation_split=FIXED_VALIDATION,
            verbose=1
        )
        
        # --- Time tracking ends ---
        duration = time.time() - start_time

        # Save Model and Scaler
        self.model.save(MODEL_SAVE_PATH)
        with open(SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        return history, duration

    def predict(self, train_df, test_df):
        """
        Makes predictions on the test data using the saved model.

        Args:
            train_df (pd.DataFrame): The training dataframe (used for context).
            test_df (pd.DataFrame): The testing dataframe.

        Returns:
            dict: A dictionary containing original test values, predicted values, and evaluation metrics.
        """
        # --- Load Model and Scaler ---
        self.model = load_model(MODEL_SAVE_PATH)
        with open(SCALER_SAVE_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

        LOOK_BACK = FIXED_LOOK_BACK

        # --- Prepare data for prediction ---
        train_close = train_df['Close'].values.reshape(-1, 1)
        test_close = test_df['Close'].values.reshape(-1, 1)

        # Concatenate training's last LOOK_BACK days with test data
        final_df_combined = np.concatenate((train_close[-LOOK_BACK:], test_close), axis=0)
        input_data_scaled = self.scaler.transform(final_df_combined)

        x_test, y_test = [], []
        for i in range(LOOK_BACK, len(input_data_scaled)):
            x_test.append(input_data_scaled[i - LOOK_BACK:i])
            y_test.append(input_data_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Reshape x_test for LSTM input
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # --- Predict & Inverse Transform ---
        y_pred_scaled = self.model.predict(x_test)
        
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_original = self.scaler.inverse_transform(y_pred_scaled)

        # --- Evaluate Metrics ---
        mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
        mse_scaled = mean_squared_error(y_test, y_pred_scaled)
        rmse_scaled = np.sqrt(mse_scaled)
        r2_original = r2_score(y_test_original, y_pred_original)
        
        metrics_data = {
            "MAE": mae_scaled,
            "MSE": mse_scaled,
            "RMSE": rmse_scaled,
            "R2_Score": r2_original
        }

        return {
            "y_test_original": y_test_original,
            "y_pred_original": y_pred_original,
            "metrics": metrics_data
        }

    def predict_future(self, train_df, test_df, num_prediction_days=10):
        """
        Predicts future stock prices recursively.

        Args:
            train_df (pd.DataFrame): Training dataframe.
            test_df (pd.DataFrame): Testing dataframe.
            num_prediction_days (int): Number of future days to predict.

        Returns:
            list: A list of predicted prices for the future days.
        """
        LOOK_BACK = FIXED_LOOK_BACK
        
        if 'Close' not in train_df.columns or 'Close' not in test_df.columns:
            raise ValueError("Missing 'Close' column in data.")

        # Combine training and testing data for full history
        combined_data = np.concatenate((train_df['Close'].values, test_df['Close'].values), axis=0).reshape(-1, 1)
        
        if len(combined_data) < LOOK_BACK:
             raise ValueError(f"Not enough data for LOOK_BACK={LOOK_BACK}")

        # Get the last LOOK_BACK days to start prediction
        current_sequence = combined_data[-LOOK_BACK:].copy()
        future_predictions = []

        for _ in range(num_prediction_days):
            # 1. Scale the current input sequence
            scaled_current_sequence = self.scaler.transform(current_sequence)
            # 2. Reshape for model prediction
            reshaped_input = np.reshape(scaled_current_sequence, (1, LOOK_BACK, 1))
            
            # 3. Make prediction
            next_day_prediction_scaled = self.model.predict(reshaped_input, verbose=0)
            # 4. Inverse transform the prediction
            next_day_prediction_original = self.scaler.inverse_transform(next_day_prediction_scaled)

            future_predictions.append(next_day_prediction_original[0][0])

            # 5. Update the input sequence for the next prediction (Recursive Step)
            current_sequence = np.concatenate((current_sequence[1:], next_day_prediction_original), axis=0)
            
        return future_predictions

