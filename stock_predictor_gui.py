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
    def __init__(self, master, on_back_callback=None): # Added on_back_callback parameter
        """
        Initializes the Tkinter application.
        Sets up the main window, variables, and calls the widget creation method.

        Args:
            master (tk.Tk): The root Tkinter window.
            on_back_callback (callable, optional): A function to call when the "Back"
                                                   button is pressed, typically to return
                                                   to a previous screen (e.g., welcome page).
        """
        self.master = master
        self.on_back_callback = on_back_callback # Store the callback
        master.title("Stock Price Prediction System")
        # Set initial geometry to a 3:2 width:height ratio, e.g., 1050x700
        initial_width = 1050
        initial_height = 700
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width // 2) - (initial_width // 2)
        y = (screen_height // 2) - (initial_height // 2)
        master.geometry(f"{initial_width}x{initial_height}+{x}+{y}")
        master.resizable(True, True) # Allow resizing the window and use OS's maximize button

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

        # Store evaluation metrics for plotting
        self.metrics_data = {
            "MAE": None,
            "MSE": None,
            "RMSE": None,
            "R2_Score": None
        }

        # --- Create GUI Widgets ---
        self.create_widgets()

        # --- Initial check for saved model/scaler ---
        # This updates the status bar and sets initial states of plot buttons.
        self.check_saved_model_status()

    def create_widgets(self):
        """
        Creates and arranges all the GUI elements (buttons, labels, text areas, etc.).
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

        # --- Frame for Output and Plots ---
        output_frame = tk.LabelFrame(self.master, text="Prediction Results & Plots", padx=10, pady=10)
        # Pack to fill horizontally and expand vertically
        output_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # --- First Button Control Frame: Predict, Show Model Loss, Show Prediction Plot ---
        first_button_control_frame = tk.Frame(output_frame, padx=10, pady=5)
        first_button_control_frame.pack(pady=5) # Pack this frame above the output_text

        # Configure columns to expand equally for buttons in the first frame (now 4 columns)
        first_button_control_frame.grid_columnconfigure(0, weight=1)
        first_button_control_frame.grid_columnconfigure(1, weight=1)
        first_button_control_frame.grid_columnconfigure(2, weight=1)
        first_button_control_frame.grid_columnconfigure(3, weight=1) # New column for the fourth button

        self.predict_10_days_button = tk.Button(first_button_control_frame, text=" Predict Next 10 Days", command=self.start_future_prediction_thread, bg="#FFC107" \
        "", fg="black", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.predict_10_days_button.grid(row=0, column=0, padx=5, pady=5)

        # NEW BUTTON: Plot Evaluation Metrics
        self.metrics_plot_button = tk.Button(first_button_control_frame, text=" Show Metrics Plot", command=self.show_evaluation_metrics_plot, bg="#FFC107", fg="black", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.metrics_plot_button.grid(row=0, column=1, padx=5, pady=5) # Placed at column 1

        self.loss_plot_button = tk.Button(first_button_control_frame, text="📈 Show Model Loss", command=self.show_loss_plot, bg="#FFC107", fg="black", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.loss_plot_button.grid(row=0, column=2, padx=5, pady=5) # Shifted to column 2

        self.prediction_plot_button = tk.Button(first_button_control_frame, text="📈 Show Prediction Plot", command=self.show_prediction_plot, bg="#FFC107", fg="black", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.prediction_plot_button.grid(row=0, column=3, padx=5, pady=5) # Shifted to column 3

        # ScrolledText widget for displaying messages and evaluation metrics
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=70, height=10, state='disabled', font=("Consolas", 10))
        self.output_text.pack(pady=5, padx=5, fill="both", expand=True) # This is now between the two button frames

        # --- Second Button Control Frame: Previous, Clear Output, Exit ---
        second_button_control_frame = tk.Frame(output_frame, padx=10, pady=5)
        second_button_control_frame.pack(pady=5) # Pack this frame below the output_text

        # Configure columns to expand equally for buttons in the second frame
        second_button_control_frame.grid_columnconfigure(0, weight=1)
        second_button_control_frame.grid_columnconfigure(1, weight=1)
        second_button_control_frame.grid_columnconfigure(2, weight=1)

        self.back_button = tk.Button(second_button_control_frame, text="<<Home", command=self.back_to_welcome, bg="#607D8B", fg="white", font=("Arial", 10, "bold"))
        self.back_button.grid(row=0, column=0, padx=5, pady=5)

        self.clear_button = tk.Button(second_button_control_frame, text=" Clear Output", command=self.clear_output, bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        self.clear_button.grid(row=0, column=1, padx=5, pady=5)

        # Corrected spelling from "Exist" to "Exit"
        self.end_button = tk.Button(second_button_control_frame, text="Exit", command=self.end_application, bg="#E91E63", fg="white", font=("Arial", 10, "bold"))
        self.end_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Status Bar ---
        # Label at the bottom to show application status
        self.status_label = tk.Label(self.master, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 9))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # Removed maximize_window and restore_window methods
    # These functionalities are typically handled by the operating system's window controls
    # if master.resizable(True, True) is set.

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
                self.update_output(f"Loaded  Dataset For Training: {os.path.basename(file_path)}. Now Training Model...")
                self.update_status("Training data loaded. Starting model training...")
                self.start_training_thread() # Automatically start training in a new thread
            except Exception as e:
                print(f"Error loading training CSV: {e}") # Debugging print
                messagebox.showerror("Error", f"Failed to load training CSV: {e}")
                self.update_status("Failed to load training data.")
                self.train_df = None # Reset dataframe on error
                self.loss_plot_button.config(state=tk.DISABLED)
                self.prediction_plot_button.config(state=tk.DISABLED)
                self.predict_10_days_button.config(state=tk.DISABLED) # Disable new button on error
                self.metrics_plot_button.config(state=tk.DISABLED) # Disable new button on error

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
                self.update_output(f"Loaded  Dataset For Testing: {os.path.basename(file_path)}.")
                self.update_status("Testing data loaded. Starting prediction...")
                self.load_and_predict() # Automatically start prediction
                # Enable the new 10-day prediction button after successful test data load and prediction
                self.predict_10_days_button.config(state=tk.NORMAL) 
            except Exception as e:
                print(f"Error loading testing CSV: {e}") # Debugging print
                messagebox.showerror("Error", f"Failed to load testing CSV: {e}")
                self.update_status("Failed to load testing data.")
                self.test_df = None # Reset dataframe on error
                self.loss_plot_button.config(state=tk.DISABLED)
                self.prediction_plot_button.config(state=tk.DISABLED)
                self.predict_10_days_button.config(state=tk.DISABLED) # Disable new button on error
                self.metrics_plot_button.config(state=tk.DISABLED) # Disable new button on error

    def start_training_thread(self):
        """
        Starts the model training process in a separate thread to prevent the GUI from freezing.
        Disables plot buttons during training.
        """
        if self.train_df is None:
            messagebox.showwarning("Missing Data", "Training data is not loaded. Cannot start training.")
            return

        self.update_status("Training Model... (This may take a while)")
        #self.update_output("Training model...") # Initial message for training start
        # Disable plot buttons immediately when training starts
        self.loss_plot_button.config(state=tk.DISABLED)
        self.prediction_plot_button.config(state=tk.DISABLED)
        self.predict_10_days_button.config(state=tk.DISABLED) # Disable new button during training
        self.metrics_plot_button.config(state=tk.DISABLED) # Disable new button during training

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
            #early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)
            # Train Model
            #self.update_output(f"Starting model training with {EPOCHS} epochs and batch size {BATCH_SIZE}...")
            self.history = self.model.fit(x_train, y_train,
                                            epochs=EPOCHS,
                                            batch_size=BATCH_SIZE,
                                            callbacks=[early_stop], #that's auto stop function
                                            validation_split=0.2, # Use 20% of training data for validation
                                            verbose=1) # Set verbose to 1 to show training progress in console
            # Save Model and Scaler after successful training
            self.model.save(MODEL_SAVE_PATH)
            with open(SCALER_SAVE_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)

            self.update_output("Training Model Successfully...")
            self.update_status("Model training complete and saved.")
            self.master.after(0, lambda: self.loss_plot_button.config(state=tk.NORMAL)) # Enable loss plot button after training
            # Prediction plot button and 10-day prediction button will be enabled after actual prediction (test CSV load)
            
        except Exception as e:
            print(f"Error during training: {e}") # Debugging print
            self.update_output(f"An error occurred during training: {e}") # Display error in output
            messagebox.showerror("Training Error", f"An error occurred during training: {e}")
            self.update_status("Training failed.")
            # Disable plot buttons if training failed
            self.master.after(0, lambda: self.loss_plot_button.config(state=tk.DISABLED))
            self.master.after(0, lambda: self.prediction_plot_button.config(state=tk.DISABLED))
            self.master.after(0, lambda: self.predict_10_days_button.config(state=tk.DISABLED))
            self.master.after(0, lambda: self.metrics_plot_button.config(state=tk.DISABLED))

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
        #self.update_output("Loading model and scaler...")

        try:
            # Load Model
            self.model = load_model(MODEL_SAVE_PATH)
            # Load Scaler
            with open(SCALER_SAVE_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            #self.update_output("Model and scaler loaded successfully from local files.")

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
            #self.update_output("Making predictions on test data...")
            y_pred_scaled = self.model.predict(x_test)

            # Inverse transform both predictions and actual test values to original scale
            # This is crucial for interpreting the results in the original price range.
            self.y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            self.y_pred_original = self.scaler.inverse_transform(y_pred_scaled)

            # Evaluate on Scaled Data (MAE, MSE, RMSE) and Original Data (R2)
            mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
            mse_scaled = mean_squared_error(y_test, y_pred_scaled) # Added MSE calculation
            rmse_scaled = np.sqrt(mse_scaled)

            r2_original = r2_score(self.y_test_original, self.y_pred_original)

            # Calculate Mean Absolute Percentage Error (MAPE)
            def mean_absolute_percentage_error(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                # Avoid division by zero for true values that are zero
                non_zero_true = y_true != 0
                if not np.any(non_zero_true):
                    return np.nan # Return NaN if all true values are zero
                return np.mean(np.abs((y_true[non_zero_true] - y_pred[non_zero_true]) / y_true[non_zero_true])) 
#               return np.mean(np.abs((y_true[non_zero_true] - y_pred[non_zero_true]) / y_true[non_zero_true])) * 100

            mape_original = mean_absolute_percentage_error(self.y_test_original, self.y_pred_original)
            #accuracy_percentage = 100 - mape_original if mape_original is not np.nan else np.nan
            accuracy_decimal= 1 - mape_original if mape_original is not np.nan else np.nan
            #acuracy_baseOnZero=accuracy_percentage/100

            # Store metrics for plotting
            self.metrics_data["MAE"] = mae_scaled
            self.metrics_data["MSE"] = mse_scaled
            self.metrics_data["RMSE"] = rmse_scaled
            self.metrics_data["R2_Score"] = r2_original
            # Optional: if you want to plot MAPE/Accuracy, store them too
            # self.metrics_data["MAPE"] = mape_original 
            # self.metrics_data["Accuracy"] = accuracy_decimal


            # Display results in the scrolled text box
            self.update_output("\n--- Evaluation Metrics On Testing---")
            self.update_output(f"📊 MAE    : {mae_scaled:.4f}")
            self.update_output(f"📊 MSE    : {mse_scaled:.4f}") # Display MSE
            self.update_output(f"📊 RMSE   : {rmse_scaled:.4f}")
            self.update_output(f"📊 R² Score:{r2_original:.4f}")
            #self.update_output(f"🎯 Prediction Accuracy: {accuracy_decimal:.4f}")
            
            # --- Display the entire testing dataset with robust column check ---
            self.update_output("\n--- Testing Data (date and close Price) ---")
            try:
                required_columns = ['date', 'close']
                if all(col in self.test_df.columns for col in required_columns):
                    all_test_data = self.test_df[required_columns]
                    self.update_output(all_test_data.to_string(index=False))
                else:
                    self.update_output("Warning: The 'date' or 'close' column is missing from the testing data file.")

            except Exception as e:
                self.update_output(f"Could not display all testing data. Error: {e}")
            # --- End of all testing data display ---

            self.update_status("Prediction complete. Results displayed.")
            # Enable both plot buttons after successful prediction
            self.master.after(0, lambda: self.loss_plot_button.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.prediction_plot_button.config(state=tk.NORMAL))
            # The 10-day prediction button is enabled here, after test data is loaded and initial prediction is done.
            self.master.after(0, lambda: self.predict_10_days_button.config(state=tk.NORMAL)) 
            self.master.after(0, lambda: self.metrics_plot_button.config(state=tk.NORMAL)) # Enable new metrics plot button

        except FileNotFoundError:
            print(f"Error: Model or scaler files not found. Please train the model first. {MODEL_SAVE_PATH}, {SCALER_SAVE_PATH}") # Debugging print
            messagebox.showerror("Error", "Saved model or scaler not found. Please train the model first by loading a training CSV.")
            self.update_status("Prediction failed: Model not found.")
            # Disable plot buttons if files are missing
            self.master.after(0, lambda: self.loss_plot_button.config(state=tk.DISABLED))
            self.master.after(0, lambda: self.prediction_plot_button.config(state=tk.DISABLED))
            self.master.after(0, lambda: self.predict_10_days_button.config(state=tk.DISABLED)) # Disable new button on error
            self.master.after(0, lambda: self.metrics_plot_button.config(state=tk.DISABLED)) # Disable new button on error
        except Exception as e:
            print(f"Error during prediction: {e}") # Debugging print
            self.update_output(f"An error occurred during prediction: {e}") # Display error in output
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.update_status("Prediction failed.")
            # Disable plot buttons if prediction fails
            self.master.after(0, lambda: self.loss_plot_button.config(state=tk.DISABLED))
            self.master.after(0, lambda: self.prediction_plot_button.config(state=tk.DISABLED))
            self.master.after(0, lambda: self.predict_10_days_button.config(state=tk.DISABLED)) # Disable new button on error
            self.master.after(0, lambda: self.metrics_plot_button.config(state=tk.DISABLED)) # Disable new button on error

    def start_future_prediction_thread(self):
        """
        Starts the 10-day future prediction process in a separate thread.
        Disables the button during prediction.
        """
        if self.model is None or self.scaler is None or self.train_df is None or self.test_df is None:
            messagebox.showwarning("Prerequisites Missing", "Please load both training and testing data first to train/load the model and scaler.")
            return

        self.update_status("Predicting next 10 days... (This may take a moment)")
        self.update_output("\n--- Predict Next 10-Days ---")
        self.predict_10_days_button.config(state=tk.DISABLED) # Disable button during prediction

        prediction_thread = threading.Thread(target=self._predict_next_10_days)
        prediction_thread.daemon = True
        prediction_thread.start()

    def _predict_next_10_days(self):
        """
        Performs recursive 10-day future prediction using the trained model.
        This method runs in a separate thread.
        """
        try:
            LOOK_BACK = FIXED_LOOK_BACK
            num_prediction_days = 10

            # Combine training and testing data to get the full historical sequence
            # Ensure 'close' column exists in both dataframes
            if 'close' not in self.train_df.columns or 'close' not in self.test_df.columns:
                raise ValueError("Missing 'close' column in either training or testing data.")

            combined_data = np.concatenate((self.train_df['close'].values, self.test_df['close'].values), axis=0).reshape(-1, 1)

            # Get the last LOOK_BACK days from this combined data to start the prediction
            if len(combined_data) < LOOK_BACK:
                self.update_output(f"Error: Not enough combined data points ({len(combined_data)}) for LOOK_BACK={LOOK_BACK} to start future prediction.")
                self.update_status("Future prediction aborted: Insufficient data.")
                return

            current_sequence = combined_data[-LOOK_BACK:].copy()
            future_predictions_original = []

            for day in range(num_prediction_days):
                # 1. Scale the current input sequence
                scaled_current_sequence = self.scaler.transform(current_sequence)

                # 2. Reshape for model prediction: (1, LOOK_BACK, 1)
                reshaped_input = np.reshape(scaled_current_sequence, (1, LOOK_BACK, 1))

                # 3. Make prediction for the next day (output will be scaled)
                next_day_prediction_scaled = self.model.predict(reshaped_input, verbose=0)

                # 4. Inverse transform the prediction back to its original price scale
                next_day_prediction_original = self.scaler.inverse_transform(next_day_prediction_scaled)

                # Store the original price prediction
                future_predictions_original.append(next_day_prediction_original[0][0])

                # 5. Update the input sequence for the next prediction (Recursive Step)
                # Remove the oldest data point from the sequence
                # Append the newly predicted price (after reshaping it to 2D for concatenation)
                current_sequence = np.concatenate((current_sequence[1:], next_day_prediction_original.reshape(-1, 1)), axis=0)
                
                self.update_output(f"Predicted Day {day + 1}: {next_day_prediction_original[0][0]:.2f}")

            #self.update_output("\n--- 10-Day Forecast Complete ---")
            self.update_status("10-day future prediction complete.")

        except Exception as e:
            print(f"Error during 10-day future prediction: {e}") # Debugging print
            self.update_output(f"An error occurred during 10-day future prediction: {e}")
            messagebox.showerror("Prediction Error", f"An error occurred during 10-day future prediction: {e}")
            self.update_status("10-day future prediction failed.")
        finally:
            self.master.after(0, lambda: self.predict_10_days_button.config(state=tk.NORMAL)) # Re-enable button

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

    def show_evaluation_metrics_plot(self):
        """
        Displays a bar chart of the evaluation metrics (MAE, MSE, RMSE, R2 Score)
        in a new Tkinter window.
        """
        # Ensure metrics_data is populated
        if not all(self.metrics_data.values()):
            messagebox.showwarning("No Data", "Evaluation metrics have not been calculated yet. Please load test data for prediction.")
            return

        metrics_window = tk.Toplevel(self.master)
        metrics_window.title("Evaluation Metrics Overview")
        metrics_window.geometry("700x500")

        metrics_names = list(self.metrics_data.keys())
        metrics_values = list(self.metrics_data.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Define colors for the bars
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        bars = ax.bar(metrics_names, metrics_values, color=colors)
        
        ax.set_title("📊 Model Evaluation Metrics")
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(metrics_values) * 1.2 if metrics_values else 1) # Adjust y-axis limit
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (max(metrics_values) * 0.02 if metrics_values else 0.02), 
                    round(yval, 4), ha='center', va='bottom')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=metrics_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, metrics_window)
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

        # Reset stored metrics
        self.metrics_data = {
            "MAE": None,
            "MSE": None,
            "RMSE": None,
            "R2_Score": None
        }
        
        # Re-check status of saved model files (this will disable plot buttons and new prediction button)
        self.check_saved_model_status() 

    def check_saved_model_status(self):
        """
        Checks if a saved model and scaler exist and updates the status bar and plot buttons accordingly.
        This is called on app startup and after clearing output.
        """
        model_exists = os.path.exists(MODEL_SAVE_PATH)
        scaler_exists = os.path.exists(SCALER_SAVE_PATH)

        if model_exists and scaler_exists:
            self.update_status("Ready to load data and predict.")
        else:
            self.update_status("No saved model/scaler found. Please load training data to begin.")
        
        # Always disable plot buttons and new prediction button on initial check or clear,
        # they get enabled after successful operations.
        self.loss_plot_button.config(state=tk.DISABLED)
        self.prediction_plot_button.config(state=tk.DISABLED)
        self.predict_10_days_button.config(state=tk.DISABLED)
        self.metrics_plot_button.config(state=tk.DISABLED) # Disable new button too

    def back_to_welcome(self):
        """
        Hides the current StockPredictorApp window and calls the callback
        to return to the welcome page.
        """
        if self.on_back_callback:
            # First, destroy any open plot windows to prevent them from lingering
            for widget in self.master.winfo_children():
                # Check for Toplevel windows by type or title
                if isinstance(widget, tk.Toplevel) and ("plot" in widget.title().lower() or "prediction" in widget.title().lower() or "loss" in widget.title().lower() or "metrics" in widget.title().lower()):
                    widget.destroy() # Close plot windows
            
            self.master.withdraw() # Hide the main app window
            self.on_back_callback() # Call the function to show the welcome page
        else:
            messagebox.showwarning("Navigation Error", "No callback function provided to go back to the welcome page.")
        
    def end_application(self):
        """
        Confirms with the user and then terminates the entire Tkinter application.
        """
        if messagebox.askyesno("Exit Application", "Are you sure you want to exit the application?"):
            # Destroy all Toplevel windows first (like plot windows)
            for widget in self.master.winfo_children():
                if isinstance(widget, tk.Toplevel):
                    widget.destroy()
            self.master.quit() # Quits the mainloop and terminates the application

# --- Main execution block ---
if __name__ == "__main__":
    # Create the root Tkinter window
    root = tk.Tk()
    # For standalone testing, define a dummy callback
    def dummy_back_callback():
        print("Dummy back callback: Would normally show welcome page.")
        # In a real scenario, you might re-create and show the WelcomePage here
        # For example: WelcomePage(root, start_main_app_callback).deiconify()
    
    # Instantiate the StockPredictorApp, passing the dummy callback
    app = StockPredictorApp(root, on_back_callback=dummy_back_callback)
    # Start the Tkinter event loop
    root.mainloop()

