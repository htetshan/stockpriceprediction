import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import threading

class LSTM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LSTM Trainer")
        self.root.geometry("500x450")

        self.train_file_path = tk.StringVar()
        self.test_file_path = tk.StringVar()

        # Training Data Selection
        tk.Label(root, text="Select Training Dataset:").pack()
        tk.Entry(root, textvariable=self.train_file_path, width=40).pack()
        tk.Button(root, text="Browse", command=self.load_train_file).pack()
        tk.Button(root, text="Load Data", command=self.load_train_data).pack()

        # Testing Data Selection
        tk.Label(root, text="Select Testing Dataset:").pack()
        tk.Entry(root, textvariable=self.test_file_path, width=40).pack()
        tk.Button(root, text="Browse", command=self.load_test_file).pack()
        tk.Button(root, text="Test Data", command=self.test_model).pack()

        # Train Button
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack()

        # Metrics Display
        self.rmse_var = tk.StringVar()
        self.mse_var = tk.StringVar()
        self.r2_var = tk.StringVar()
        
        tk.Label(root, text="RMSE:").pack()
        tk.Entry(root, textvariable=self.rmse_var, state='readonly').pack()
        tk.Label(root, text="MSE:").pack()
        tk.Entry(root, textvariable=self.mse_var, state='readonly').pack()
        tk.Label(root, text="R² Score:").pack()
        tk.Entry(root, textvariable=self.r2_var, state='readonly').pack()

        # Clear Button
        tk.Button(root, text="Clear", command=self.clear_all).pack()

        self.data = None
        self.test_data = None
        self.model = None
        self.scaler = MinMaxScaler()

    def load_train_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.train_file_path.set(file_path)
    
    def load_test_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.test_file_path.set(file_path)
    
    def load_train_data(self):
        if not self.train_file_path.get():
            messagebox.showerror("Error", "Please select a training dataset")
            return
        try:
            self.data = pd.read_csv(self.train_file_path.get())

            # Check if first column is a date and convert it
            if isinstance(self.data.iloc[0, 0], str) and "-" in self.data.iloc[0, 0]:
                self.data.iloc[:, 0] = pd.to_datetime(self.data.iloc[:, 0])
                self.data.iloc[:, 0] = self.data.iloc[:, 0].map(pd.Timestamp.toordinal)
            
            self.data = self.data.astype(float)
            messagebox.showinfo("Success", "Training data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load training data: {e}")
    
    def train_model(self):
        if self.data is None:
            messagebox.showerror("Error", "Load training data first")
            return
        
        self.train_button.config(text="Training...", state=tk.DISABLED)
        threading.Thread(target=self.run_training, daemon=True).start()
    
    def run_training(self):
        try:
            data = self.data.values.astype(float)
            data = self.scaler.fit_transform(data)
            X, y = data[:-1], data[1:, 0]  # Simple LSTM model structure
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            self.model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50, activation='relu'),  # The last LSTM layer should not return sequences
                 Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse')
            
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=1)
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")
        finally:
            self.train_button.config(text="Train Model", state=tk.NORMAL)
    
    def test_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Train the model first")
            return
        
        if not self.test_file_path.get():
            messagebox.showerror("Error", "Please select a test dataset")
            return
        try:
            test_data = pd.read_csv(self.test_file_path.get())
            
            # Check if first column is a date and convert it
            if isinstance(test_data.iloc[0, 0], str) and "-" in test_data.iloc[0, 0]:
                test_data.iloc[:, 0] = pd.to_datetime(test_data.iloc[:, 0])
                test_data.iloc[:, 0] = test_data.iloc[:, 0].map(pd.Timestamp.toordinal)
            
            test_data = test_data.astype(float)
            test_data = self.scaler.transform(test_data)
            X_test, y_test = test_data[:-1], test_data[1:, 0]
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            y_pred = self.model.predict(X_test)
            y_pred = self.scaler.inverse_transform(np.concatenate([np.zeros((y_pred.shape[0], X_test.shape[1] - 1)), y_pred], axis=1))[:, -1]
            y_test = self.scaler.inverse_transform(np.concatenate([np.zeros((y_test.shape[0], X_test.shape[1] - 1)), y_test.reshape(-1, 1)], axis=1))[:, -1]
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.rmse_var.set(f"{rmse:.4f}")
            self.mse_var.set(f"{mse:.4f}")
            self.r2_var.set(f"{r2:.4f}")
            
            plt.figure(figsize=(8, 5))
            plt.plot(y_test, label="Actual")
            plt.plot(y_pred, label="Predicted")
            plt.legend()
            plt.title("Predicted vs Actual Values")
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {e}")
    
    def clear_all(self):
        self.train_file_path.set("")
        self.test_file_path.set("")
        self.rmse_var.set("")
        self.mse_var.set("")
        self.r2_var.set("")
        self.data = None
        self.test_data = None
        self.model = None
        messagebox.showinfo("Clear", "All fields and data have been cleared!")

if __name__ == "__main__":
    root = tk.Tk()
    app = LSTM_GUI(root)
    root.mainloop()
