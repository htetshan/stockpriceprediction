import tkinter as tk
from tkinter import filedialog, scrolledtext
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

class LSTMModelGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSTM Model Training")
        self.geometry("800x500")

        self.train_data = None
        self.test_data = None
        
        # Create Buttons
        self.load_train_button = tk.Button(self, text="Load Training Data", command=self.load_train_data)
        self.load_train_button.pack(pady=5)

        self.load_test_button = tk.Button(self, text="Load Testing Data", command=self.load_test_data)
        self.load_test_button.pack(pady=5)

        self.train_button = tk.Button(self, text="Train LSTM Model", command=self.train_model)
        self.train_button.pack(pady=5)

        self.clear_button = tk.Button(self, text="Clear", command=self.clear_all, bg="red", fg="white")
        self.clear_button.pack(pady=5)

        # ScrolledText for results
        self.result_text = scrolledtext.ScrolledText(self, width=70, height=15)
        self.result_text.pack(pady=10)

    def load_train_data(self):
        """ Load Training Data from File """
        file_path = filedialog.askopenfilename(title="Select Training Data", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.train_data = pd.read_csv(file_path)
            self.result_text.insert(tk.END, f"Loaded training data from: {file_path}\n")

    def load_test_data(self):
        """ Load Testing Data from File """
        file_path = filedialog.askopenfilename(title="Select Testing Data", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.test_data = pd.read_csv(file_path)
            self.result_text.insert(tk.END, f"Loaded testing data from: {file_path}\n")

    def train_model(self):
        """ Train the LSTM model """
        if self.train_data is None or self.test_data is None:
            self.result_text.insert(tk.END, "Please load both training and testing data before training.\n")
            return
        
        # Prepare the training and testing data
        train_close = self.train_data.iloc[:, 4:5].values
        test_close = self.test_data.iloc[:, 4:5].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(train_close)
        
        x_train, y_train = [], []
        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100: i])
            y_train.append(data_training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Build LSTM Model
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        model.fit(x_train, y_train, epochs=100)

        # Prepare the test data
        past_100_days = pd.DataFrame(train_close[-100:])
        test_df = pd.DataFrame(test_close)
        final_df = pd.concat([past_100_days, test_df], ignore_index=True)
        
        input_data = scaler.fit_transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predict and evaluate the model
        y_pred = model.predict(x_test)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        self.result_text.insert(tk.END, f"Training Complete.\nMAE: {mae}, RMSE: {rmse}, R2 Score: {r2}\n")

    def clear_all(self):
        """ Clear all fields and reset application """
        self.train_data = None
        self.test_data = None
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "All fields cleared.\n")

# Create and run the Tkinter application
app = LSTMModelGUI()
app.mainloop()
