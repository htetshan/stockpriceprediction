import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Load the trained model
try:
    model = tf.keras.models.load_model("stock_model.h5")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model: {e}")
    exit()

# Load historical data
try:
    df = pd.read_csv("google.csv")  # Ensure this file exists
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load historical data: {e}")
    exit()

# Function to predict stock prices
def predict_prices():
    try:
        days = int(entry_days.get())
        if days <= 0:
            messagebox.showwarning("Warning", "Please enter a positive number of days.")
            return
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=days+1, freq='D')[1:]
        
        # Generate dummy sequential data for prediction
        x_future = np.arange(1, days + 1).reshape(-1, 1)  # Reshape to fit model input
        predictions = model.predict(x_future)
        
        # Plot results - Full historical + future data
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df['Close'], label='Historical Close Price', color='blue')
        ax.plot(future_dates, predictions, linestyle='-', label='Predicted Close Price', color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.set_title("Stock Price Forecast")
        ax.legend()
        ax.grid()
        
        # Display graph in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # Second graph - Only future predictions with days as x-axis
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(range(1, days + 1), predictions, linestyle='-', marker='o', color='green', label=f'{days}-Day Forecast')
        ax2.set_xlabel("Days")
        ax2.set_ylabel("Close Price")
        ax2.set_title(f"{days}-Day Stock Price Prediction")
        ax2.legend()
        ax2.grid()
        
        # Display second graph in Tkinter window
        canvas2 = FigureCanvasTkAgg(fig2, master=window)
        canvas2.draw()
        canvas2.get_tk_widget().pack()
    
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid integer for days.")

# GUI setup
window = tk.Tk()
window.title("Stock Price Predictor")
window.geometry("600x800")

tk.Label(window, text="Please input days:").pack()
entry_days = tk.Entry(window)
entry_days.pack()

predict_button = tk.Button(window, text="Predict Close Price", command=predict_prices)
predict_button.pack()

window.mainloop()
