import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
import threading
import time # For simulating dummy data loading/processing time

# Define paths for saving/loading the model and scaler
# The .keras extension is the recommended format for TensorFlow models
MODEL_SAVE_PATH = 'my_lstm_model.keras'
SCALER_SAVE_PATH = 'my_scaler.pkl'

# --- Fixed Hyperparameters (No longer user-editable) ---
# These values are set to provide a reasonable starting point for the model.
FIXED_LOOK_BACK = 60 # Number of previous time steps to use as input features to predict the next time step.
FIXED_EPOCHS = 150   # Number of times the learning algorithm will work through the entire training dataset.
FIXED_BATCH_SIZE = 32 # Number of samples per gradient update.

class StockPredictorApp:
    """
    A Tkinter-based application for predicting stock prices using an LSTM model.
    It allows users to load training and testing CSV data, automatically trains
    and saves the model, performs predictions, and displays evaluation metrics
    and plots.
    """
    def __init__(self, master):
        """
        Initializes the Tkinter application.
        Sets up the main window, variables, and calls the widget creation method.

        Args:
            master (tk.Tk): The root Tkinter window.
        """
        self.master = master
        master.title("Stock Price Prediction System")
        master.geometry("800x600") # Adjusted initial window size for better layout
        master.resizable(True, True) # Allow resizing the window

        # --- Variables to store data, model, scaler, results ---
        # tk.StringVar to hold file paths, allowing dynamic updates in Entry widgets
        self.train_file_path = tk.StringVar(value="")
        self.test_file_path = tk.StringVar(value="")
        
        # DataFrames to hold loaded CSV data
        self.train_df = None
        self.test_df = None
        
        # Keras model and scikit-learn scaler objects
        self.model = None
        self.scaler = None
        
        # Stores training history for the loss plot (from model.fit)
        self.history = None
        
        # Stores original (unscaled) test actual and predicted values for plotting and evaluation
        self.y_test_original = None
        self.y_pred_original = None

        # --- Create GUI Widgets ---
        self.create_widgets()

        # --- Initial check for saved model/scaler ---
        # This updates the status bar and sets initial states of plot buttons.
        self.check_saved_model_status()

    def create_widgets(self):
        """
        Creates and arranges all the GUI elements (buttons, labels, text areas, etc.).
        The "Train & Save Model" and "Load & Predict" buttons are removed
        as their actions will be automated upon file selection.
        """
        # --- Frame for File Loading ---
        # LabelFrame provides a titled border around related widgets.
        file_frame = tk.LabelFrame(self.master, text="Data Loading", padx=10, pady=10)
        file_frame.pack(pady=10, padx=10, fill="x") # Pack to fill horizontally

        # Train CSV selection row
        tk.Label(file_frame, text="Train CSV:").grid(row=0, column=0, sticky="w", pady=2)
        # Entry widget to display selected file path (readonly as user selects via browse button)
        tk.Entry(file_frame, textvariable=self.train_file_path, width=60, state='readonly').grid(row=0, column=1, padx=5, pady=2)
        tk.Button(file_frame, text="Browse", command=self.load_train_csv).grid(row=0, column=2, padx=5, pady=2)

        # Test CSV selection row
        tk.Label(file_frame, text="Test CSV:").grid(row=1, column=0, sticky="w", pady=2)
        tk.Entry(file_frame, textvariable=self.test_file_path, width=60, state='readonly').grid(row=1, column=1, padx=5, pady=2)
        tk.Button(file_frame, text="Browse", command=self.load_test_csv).grid(row=1, column=2, padx=5, pady=2)

        # --- Frame for Action Buttons (only Clear and Plot buttons remain) ---
        action_frame = tk.Frame(self.master, padx=10, pady=10)
        action_frame.pack(pady=10, padx=10, fill="x")

        # Clear Output button
        self.clear_button = tk.Button(action_frame, text="🧹 Clear Output", command=self.clear_output, bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        self.clear_button.grid(row=0, column=0, padx=5, pady=5) # Adjusted column to 0

        # --- Frame for Output and Plots ---
        output_frame = tk.LabelFrame(self.master, text="Prediction Results & Plots", padx=10, pady=10)
        # Pack to fill horizontally and expand vertically
        output_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # ScrolledText widget for displaying messages and evaluation metrics
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=70, height=10, state='disabled', font=("Consolas", 10))
        self.output_text.pack(pady=5, padx=5, fill="both", expand=True)

        # Frame to hold plot buttons, centered below the output text
        plot_button_frame = tk.Frame(output_frame, padx=10, pady=5)
        plot_button_frame.pack(pady=5)

        # Buttons to show plots, initially disabled
        self.loss_plot_button = tk.Button(plot_button_frame, text="📈 Show Model Loss", command=self.show_loss_plot, bg="#FFC107", fg="black", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.loss_plot_button.grid(row=0, column=0, padx=5)

        self.prediction_plot_button = tk.Button(plot_button_frame, text="📈 Show Prediction Plot", command=self.show_prediction_plot, bg="#FFC107", fg="black", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.prediction_plot_button.grid(row=0, column=1, padx=5)

        # --- Status Bar ---
        # Label at the bottom to show application status
        self.status_label = tk.Label(self.master, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 9))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_output(self, message):
        """
        Inserts a message into the scrolled text output area.
        Enables the widget, inserts text, then disables it again.
        Ensures thread-safe update by using master.after if called from a non-GUI thread.
        """
        # Use master.after to ensure GUI updates happen on the main thread
        self.master.after(0, lambda: self._do_update_output(message))

    def _do_update_output(self, message):
        """Helper for update_output to perform the actual GUI update."""
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END) # Scroll to the end
        self.output_text.config(state='disabled')

    def update_status(self, message):
        """
        Updates the text in the status bar at the bottom of the window.
        Ensures thread-safe update by using master.after if called from a non-GUI thread.
        """
        self.master.after(0, lambda: self._do_update_status(message))

    def _do_update_status(self, message):
        """Helper for update_status to perform the actual GUI update."""
        self.status_label.config(text=message)
        self.master.update_idletasks() # Force update of GUI

    def load_train_csv(self):
        """
        Opens a file dialog for the user to select the training CSV.
        Loads the CSV into a pandas DataFrame and then automatically starts training.
        """
        file_path = filedialog.askopenfilename(
            title="Select Training Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.train_file_path.set(file_path)
            self.update_status(f"Loading training data from: {os.path.basename(file_path)}...")
            try:
                self.train_df = pd.read_csv(file_path)
                # Clear previous output before showing new message related to training
                self.output_text.config(state='normal')
                self.output_text.delete(1.0, tk.END)
                self.output_text.config(state='disabled')
                self.update_output(f"Loaded training data {os.path.basename(file_path)} successfully.")
                self.update_status("Training data loaded. Starting model training...")
                self.start_training_thread() # Automatically start training in a new thread
            except Exception as e:
                print(f"Error loading training CSV: {e}") # Debugging print
                messagebox.showerror("Error", f"Failed to load training CSV: {e}")
                self.update_status("Failed to load training data.")
                self.train_df = None # Reset dataframe on error
                self.loss_plot_button.config(state=tk.DISABLED)
                self.prediction_plot_button.config(state=tk.DISABLED)

    def load_test_csv(self):
        """
        Opens a file dialog for the user to select the testing CSV.
        Loads the CSV into a pandas DataFrame and then automatically starts prediction.
        Includes a check to ensure model and scaler are saved before proceeding.
        """
        # Check if model and scaler exist before proceeding to prediction
        model_exists = os.path.exists(MODEL_SAVE_PATH)
        scaler_exists = os.path.exists(SCALER_SAVE_PATH)

        if not (model_exists and scaler_exists):
            messagebox.showwarning("Model Not Found", "Please train the model first by loading a training CSV before attempting to load test data for prediction.")
            self.update_status("Prediction aborted: Model not trained.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Testing Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.test_file_path.set(file_path)
            self.update_status(f"Loading testing data from: {os.path.basename(file_path)}...")
            try:
                self.test_df = pd.read_csv(file_path)
                # Clear previous output before showing new message related to prediction
                self.output_text.config(state='normal')
                self.output_text.delete(1.0, tk.END)
                self.output_text.config(state='disabled')
                self.update_output(f"Loaded testing data {os.path.basename(file_path)} successfully.")
                self.update_status("Testing data loaded. Starting prediction...")
                self.load_and_predict() # Automatically start prediction
            except Exception as e:
                print(f"Error loading testing CSV: {e}") # Debugging print
                messagebox.showerror("Error", f"Failed to load testing CSV: {e}")
                self.update_status("Failed to load testing data.")
                self.test_df = None # Reset dataframe on error
                self.loss_plot_button.config(state=tk.DISABLED)
                self.prediction_plot_button.config(state=tk.DISABLED)

    def start_training_thread(self):
        """
        Starts the model training process in a separate thread to prevent the GUI from freezing.
        Disables plot buttons during training.
        """
        if self.train_df is None:
            messagebox.showwarning("Missing Data", "Training data is not loaded. Cannot start training.")
            return

        self.update_status("Training model... (This may take a while)")
        self.update_output("Training model...") # Initial message for training start
        # Disable plot buttons immediately when training starts
        self.loss_plot_button.config(state=tk.DISABLED)
        self.prediction_plot_button.config(state=tk.DISABLED)

        # Create and start a new thread for training
        training_thread = threading.Thread(target=self._train_model)
        training_thread.daemon = True # Allow the thread to exit with the main program
        training_thread.start()

    def _train_model(self):
        """
        Contains the core logic for training the LSTM model.
        This method runs in a separate thread.
        It handles data scaling, model building, training, and saving.
        """
        try:
            # Use fixed hyperparameters defined at the top of the script
            LOOK_BACK = FIXED_LOOK_BACK
            EPOCHS = FIXED_EPOCHS
            BATCH_SIZE = FIXED_BATCH_SIZE

            # Use only 'close' price for training
            train_close = self.train_df['close'].values.reshape(-1, 1)

            # Initialize and fit the MinMaxScaler on the training data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array_scaled = self.scaler.fit_transform(train_close)

            # Prepare Training Data: Create sequences (X) and corresponding labels (y)
            # X will be a sequence of 'LOOK_BACK' previous prices, y will be the next price
            x_train, y_train = [], []
            for i in range(LOOK_BACK, len(data_training_array_scaled)):
                x_train.append(data_training_array_scaled[i - LOOK_BACK:i])
                y_train.append(data_training_array_scaled[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape x_train for LSTM input: (samples, time_steps, features)
            # Here, time_steps is LOOK_BACK, and features is 1 (since we use only 'close' price)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build LSTM Model
            self.update_output("Building LSTM model...")
            self.model = Sequential([
                Input(shape=(x_train.shape[1], 1)), # Input shape (LOOK_BACK, 1)
                LSTM(units=100, return_sequences=True), # First LSTM layer, returns sequences for next LSTM
                Dropout(0.3), # Dropout for regularization to prevent overfitting
                LSTM(units=100, return_sequences=False), # Second LSTM layer, does not return sequences
                Dropout(0.3), # Another dropout layer
                Dense(units=1) # Output layer, predicting a single value (the next close price)
            ])

            # Compile the model with Adam optimizer and mean squared error loss
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])

            # Early Stopping: Monitor 'loss' and stop if it doesn't improve for 'patience' epochs
            early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
            
            # Train Model
            self.update_output(f"Starting model training with {EPOCHS} epochs and batch size {BATCH_SIZE}...")
            self.history = self.model.fit(x_train, y_train,
                                            epochs=EPOCHS,
                                            batch_size=BATCH_SIZE,
                                            callbacks=[early_stop],
                                            validation_split=0.2, # Use 20% of training data for validation
                                            verbose=0) # Set verbose to 0 to prevent excessive console output during GUI training

            # Save Model and Scaler after successful training
            self.model.save(MODEL_SAVE_PATH)
            with open(SCALER_SAVE_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)

            self.update_output("Training model successfully.")
            self.update_status("Model training complete and saved.")
            self.loss_plot_button.config(state=tk.NORMAL) # Enable loss plot button after training
            # Prediction plot button will be enabled after actual prediction
            
        except Exception as e:
            print(f"Error during training: {e}") # Debugging print
            self.update_output(f"An error occurred during training: {e}") # Display error in output
            messagebox.showerror("Training Error", f"An error occurred during training: {e}")
            self.update_status("Training failed.")
            # Disable plot buttons if training failed
            self.loss_plot_button.config(state=tk.DISABLED)
            self.prediction_plot_button.config(state=tk.DISABLED)


    def load_and_predict(self):
        """
        Loads a pre-trained model and scaler (if available), then performs prediction
        on the test data and displays evaluation metrics.
        This method is called automatically after loading a test CSV.
        """
        if self.test_df is None:
            messagebox.showwarning("Missing Data", "Testing data is not loaded. Cannot predict.")
            return

        self.update_status("Loading model and making predictions...")
        self.update_output("Loading model and scaler...")

        try:
            # Load Model
            self.model = load_model(MODEL_SAVE_PATH)
            # Load Scaler
            with open(SCALER_SAVE_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            self.update_output("Model and scaler loaded successfully from local files.")

            LOOK_BACK = FIXED_LOOK_BACK # Use fixed look back for prediction

            # Prepare data for prediction:
            # We need the last LOOK_BACK days from the training data to form the first sequence
            # for the test data prediction.
            train_close = self.train_df['close'].values.reshape(-1, 1)
            test_close = self.test_df['close'].values.reshape(-1, 1)

            # Concatenate training's last LOOK_BACK days with test data for proper sequence formation
            # This ensures that the first prediction in the test set uses a full LOOK_BACK history.
            final_df_combined = np.concatenate((train_close[-LOOK_BACK:], test_close), axis=0)

            # Scale the combined data using the *same scaler* fitted on training data
            input_data_scaled = self.scaler.transform(final_df_combined)

            x_test, y_test = [], []
            for i in range(LOOK_BACK, len(input_data_scaled)):
                x_test.append(input_data_scaled[i - LOOK_BACK:i])
                y_test.append(input_data_scaled[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Reshape x_test for LSTM input: (samples, time_steps, features)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Predict & Inverse Transform
            self.update_output("Making predictions on test data...")
            y_pred_scaled = self.model.predict(x_test)

            # Inverse transform both predictions and actual test values to original scale
            # This is crucial for interpreting the results in the original price range.
            self.y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            self.y_pred_original = self.scaler.inverse_transform(y_pred_scaled)

            # Evaluate on Scaled Data (MAE, RMSE) and Original Data (R2, MAPE)
            mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
            rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred_scaled))

            r2_original = r2_score(self.y_test_original, self.y_pred_original)

            # Calculate Mean Absolute Percentage Error (MAPE)
            def mean_absolute_percentage_error(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                # Avoid division by zero for true values that are zero
                non_zero_true = y_true != 0
                if not np.any(non_zero_true):
                    return np.nan # Return NaN if all true values are zero
                return np.mean(np.abs((y_true[non_zero_true] - y_pred[non_zero_true]) / y_true[non_zero_true])) * 100

            mape_original = mean_absolute_percentage_error(self.y_test_original, self.y_pred_original)

            # Display results in the scrolled text box
            self.update_output("\n--- Evaluation Metrics ---")
            self.update_output(f"📊 MAE (on scaled data): {mae_scaled:.4f}")
            self.update_output(f"📊 RMSE (on scaled data): {rmse_scaled:.4f}")
            self.update_output(f"📊 R² Score: {r2_original:.4f}")
            self.update_output(f"📊 Mean Absolute Percentage Error (MAPE): {mape_original:.4f}%")

            self.update_status("Prediction complete. Results displayed.")
            # Enable both plot buttons after successful prediction
            self.loss_plot_button.config(state=tk.NORMAL)
            self.prediction_plot_button.config(state=tk.NORMAL)

        except FileNotFoundError:
            print(f"Error: Model or scaler files not found. Please train the model first. {MODEL_SAVE_PATH}, {SCALER_SAVE_PATH}") # Debugging print
            messagebox.showerror("Error", "Saved model or scaler not found. Please train the model first by loading a training CSV.")
            self.update_status("Prediction failed: Model not found.")
            # Disable plot buttons if files are missing
            self.loss_plot_button.config(state=tk.DISABLED)
            self.prediction_plot_button.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error during prediction: {e}") # Debugging print
            self.update_output(f"An error occurred during prediction: {e}") # Display error in output
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.update_status("Prediction failed.")
            # Disable plot buttons if prediction fails
            self.loss_plot_button.config(state=tk.DISABLED)
            self.prediction_plot_button.config(state=tk.DISABLED)

    def show_loss_plot(self):
        """
        Displays the model's training loss plot in a new Tkinter window.
        """
        if self.history is None:
            messagebox.showwarning("No Data", "Model has not been trained yet or history is not available.")
            return

        loss_window = tk.Toplevel(self.master) # Create a new top-level window
        loss_window.title("Model Loss During Training")
        loss_window.geometry("800x600")

        # Create a Matplotlib figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history: # Check if validation loss is available
            ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss During Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Embed the Matplotlib figure into the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=loss_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a toolbar for zooming, panning, etc.
        toolbar = NavigationToolbar2Tk(canvas, loss_window)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def show_prediction_plot(self):
        """
        Displays the actual vs. predicted prices plot in a new Tkinter window.
        """
        if self.y_test_original is None or self.y_pred_original is None:
            messagebox.showwarning("No Data", "Predictions have not been made yet.")
            return

        prediction_window = tk.Toplevel(self.master) # Create a new top-level window
        prediction_window.title("LSTM Stock Price Prediction (Test Set)")
        prediction_window.geometry("800x600")

        # Create a Matplotlib figure and axes for the prediction plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.y_test_original, label="Actual Prices", color='blue', linewidth=2)
        ax.plot(self.y_pred_original, label="Predicted Prices", color='orange', linestyle='--', linewidth=2)
        ax.set_title("📉 LSTM Stock Price Prediction (Test Set)")
        ax.set_xlabel("Time Step (Days in Test Set)")
        ax.set_ylabel("Stock Close Price")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Embed the Matplotlib figure into the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=prediction_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a toolbar
        toolbar = NavigationToolbar2Tk(canvas, prediction_window)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def clear_output(self):
        """
        Clears the scrolled text output and resets relevant variables.
        Also disables plot buttons and re-checks the model status.
        """
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        self.update_status("Output cleared.")
        
        # Reset prediction-related data
        self.y_test_original = None
        self.y_pred_original = None
        
        # Reset training history and model/scaler objects in memory
        self.history = None
        self.model = None
        self.scaler = None
        
        # Re-check status of saved model files (this will disable plot buttons)
        self.check_saved_model_status() 

    def check_saved_model_status(self):
        """
        Checks if a saved model and scaler exist and updates the status bar and plot buttons accordingly.
        This is called on app startup and after clearing output.
        """
        model_exists = os.path.exists(MODEL_SAVE_PATH)
        scaler_exists = os.path.exists(SCALER_SAVE_PATH)

        if model_exists and scaler_exists:
            self.update_status("Saved model and scaler found. Ready to load data and predict.")
        else:
            self.update_status("No saved model/scaler found. Please load training data to begin.")
        
        # Always disable plot buttons on initial check or clear, they get enabled after successful operations
        self.loss_plot_button.config(state=tk.DISABLED)
        self.prediction_plot_button.config(state=tk.DISABLED)

# --- Main execution block ---
if __name__ == "__main__":
    # Create the root Tkinter window
    root = tk.Tk()
    # Instantiate the StockPredictorApp
    app = StockPredictorApp(root)
    # Start the Tkinter event loop
    root.mainloop()
