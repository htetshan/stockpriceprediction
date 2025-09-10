import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, font
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import threading
import time

# Import the machine learning model logic from model.py
from model import StockModel

class StockPredictorApp:
    """
    A Tkinter-based application for predicting stock prices using an LSTM model.
    It allows users to load training and testing CSV data, automatically trains
    and saves the model, performs predictions, and displays evaluation metrics
    and plots.
    """
    def __init__(self, master, on_back_callback=None):
        """
        Initializes the Tkinter application.
        Sets up the main window, variables, and calls the widget creation method.
        """
        self.master = master
        self.on_back_callback = on_back_callback
        master.title("Stock Price Prediction System")
        # Set initial geometry and center the window
        initial_width = 1200
        initial_height = 720 # Increased height slightly for better spacing
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width // 2) - (initial_width // 2)
        y = (screen_height // 2) - (initial_height // 2)
        master.geometry(f"{initial_width}x{initial_height}+{x}+{y}")
        master.resizable(True, True)

        # --- Define Fonts for a consistent and larger look ---
        self.title_font = font.Font(family="Helvetic", size=12, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=11)
        self.button_font = font.Font(family="Helvetica", size=10, weight="bold")
        self.output_font = font.Font(family="Consolas", size=13)
        self.status_font = font.Font(family="Arial", size=10)

        # --- ML Model Instance ---
        self.stock_model = StockModel()

        # --- Variables to store data, model, scaler, results ---
        self.train_file_path = tk.StringVar(value="")
        self.test_file_path = tk.StringVar(value="")
        self.train_df = None
        self.test_df = None
        self.history = None
        self.y_test_original = None
        self.y_pred_original = None
        self.test_filename_for_plot = "Test Set"
        self.train_filename_for_plot =""
        self.metrics_data = {}

        # --- Create GUI Widgets ---
        self.create_widgets()

        # --- Initial check for saved model/scaler ---
        self.check_saved_model_status()

    def create_widgets(self):
        """
        Creates and arranges all the GUI elements (buttons, labels, text areas, etc.).
        """
        # --- Frame for File Loading ---
        file_frame = tk.LabelFrame(self.master, text="Data Loading", padx=15, pady=15, font=self.title_font)
        file_frame.pack(pady=10, padx=10, fill="x")

        inner_file_frame = tk.Frame(file_frame)
        inner_file_frame.pack()

        tk.Label(inner_file_frame, text="Train CSV:", font=self.label_font).grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(inner_file_frame, textvariable=self.train_file_path, width=60, state='readonly', font=self.label_font).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(inner_file_frame, text="Browse", command=self.load_train_csv, font=self.button_font).grid(row=0, column=2, padx=10, pady=5)

        tk.Label(inner_file_frame, text="Test CSV:", font=self.label_font).grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(inner_file_frame, textvariable=self.test_file_path, width=60, state='readonly', font=self.label_font).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(inner_file_frame, text="Browse", command=self.load_test_csv, font=self.button_font).grid(row=1, column=2, padx=10, pady=5)

        # --- Frame for Output and Plots ---
        output_frame = tk.LabelFrame(self.master, text="Prediction Results & Plots", padx=15, pady=15, font=self.title_font)
        output_frame.pack(pady=10, padx=10, fill="both", expand=True)

        first_button_control_frame = tk.Frame(output_frame)
        first_button_control_frame.pack(pady=10)

        self.predict_10_days_button = tk.Button(first_button_control_frame, text="Predict Next 10 Days", command=self.start_future_prediction_thread, bg="#FFC107", fg="black", font=self.button_font, state=tk.DISABLED, padx=10, pady=5)
        self.predict_10_days_button.grid(row=0, column=0, padx=10)

        self.metrics_plot_button = tk.Button(first_button_control_frame, text="Show Metrics Plot", command=self.show_evaluation_metrics_plot, bg="#FFC107", fg="black", font=self.button_font, state=tk.DISABLED, padx=10, pady=5)
        self.metrics_plot_button.grid(row=0, column=1, padx=10)

        self.loss_plot_button = tk.Button(first_button_control_frame, text="📈 Show Model Loss", command=self.show_loss_plot, bg="#FFC107", fg="black", font=self.button_font, state=tk.DISABLED, padx=10, pady=5)
        self.loss_plot_button.grid(row=0, column=2, padx=10)

        self.prediction_plot_button = tk.Button(first_button_control_frame, text="📈 Show Prediction Plot", command=self.show_prediction_plot, bg="#FFC107", fg="black", font=self.button_font, state=tk.DISABLED, padx=10, pady=5)
        self.prediction_plot_button.grid(row=0, column=3, padx=10)

        # ScrolledText widget for displaying messages and evaluation metrics
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=70, height=18, state='disabled', font=self.output_font)
        self.output_text.pack(pady=10, padx=8, fill="both", expand=True)

        second_button_control_frame = tk.Frame(output_frame)
        second_button_control_frame.pack(pady=10)

        self.back_button = tk.Button(second_button_control_frame, text="<< Home", command=self.back_to_welcome, bg="#607D8B", fg="white", font=self.button_font, padx=10, pady=5)
        self.back_button.grid(row=0, column=0, padx=10)

        self.clear_button = tk.Button(second_button_control_frame, text="Clear Output", command=self.clear_output, bg="#f44336", fg="white", font=self.button_font, padx=10, pady=5)
        self.clear_button.grid(row=0, column=1, padx=10)

        self.end_button = tk.Button(second_button_control_frame, text="Exit", command=self.end_application, bg="#E91E63", fg="white", font=self.button_font, padx=10, pady=5)
        self.end_button.grid(row=0, column=2, padx=10)
        
        # Status Bar at the bottom
        self.status_label = tk.Label(self.master, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=self.status_font)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_output(self, message):
        """Thread-safe method to update the output text area."""
        self.master.after(0, self._do_update_output, message)

    def _do_update_output(self, message):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END) # Scroll to the end
        self.output_text.config(state='disabled')

    def update_status(self, message):
        """Thread-safe method to update the status bar."""
        self.master.after(0, self._do_update_status, message)

    def _do_update_status(self, message):
        self.status_label.config(text=message)
        self.master.update_idletasks() # Force GUI update

    def load_train_csv(self):
        """
        Opens a file dialog for the user to select the training CSV.
        Loads the CSV and automatically starts the training process.
        """
        file_path = filedialog.askopenfilename(title="Select Training Data CSV", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.train_file_path.set(file_path)
            self.train_filename_for_plot = os.path.splitext(os.path.basename(file_path))[0]
            self.update_status(f"Loading training data from: {os.path.basename(file_path)}...")
            try:
                self.train_df = pd.read_csv(file_path)
                # Clear previous output
                self.output_text.config(state='normal')
                self.output_text.delete(1.0, tk.END)
                self.output_text.config(state='disabled')
                self.update_output(f"Loaded Dataset For Training: {os.path.basename(file_path)}.\nNow Training Model...")
                self.start_training_thread()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load training CSV: {e}")
                self.update_status("Failed to load training data.")
                self.train_df = None

    def load_test_csv(self):
        """
        Opens a file dialog for the user to select the testing CSV.
        Loads the CSV and automatically starts the prediction process.
        """
        if not self.stock_model.check_model_exists():
            messagebox.showwarning("Model Not Found", "Please train the model first by loading a training CSV.")
            return

        file_path = filedialog.askopenfilename(title="Select Testing Data CSV", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.test_file_path.set(file_path)
            self.test_filename_for_plot = os.path.splitext(os.path.basename(file_path))[0]
            self.update_status(f"Loading testing data from: {os.path.basename(file_path)}...")
            try:
                self.test_df = pd.read_csv(file_path)
                self.output_text.config(state='normal')
                self.output_text.delete(1.0, tk.END)
                self.output_text.config(state='disabled')
                self.update_output(f"Loaded Dataset For Testing: {os.path.basename(file_path)}.")
                self.start_prediction_thread()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load testing CSV: {e}")
                self.update_status("Failed to load testing data.")
                self.test_df = None

    def start_training_thread(self):
        """Starts the model training process in a separate thread to prevent GUI freezing."""
        self.update_status("Training Model... (This may take a while)")
        self.set_buttons_state(tk.DISABLED)
        
        training_thread = threading.Thread(target=self._train_model_task)
        training_thread.daemon = True # Allow thread to exit with main program
        training_thread.start()

    def _train_model_task(self):
        """Task for training thread. Calls the model and updates GUI with results."""
        try:
            self.history, duration = self.stock_model.train_model(self.train_df)
            self.update_output(f"Training Model Successfully...")
            print(f"Training Model Successfully in {duration:.2f} seconds.")
            self.update_status("Model training complete and saved.")
            self.master.after(0, lambda: self.loss_plot_button.config(state=tk.NORMAL))
        except Exception as e:
            self.update_output(f"An error occurred during training: {e}")
            messagebox.showerror("Training Error", f"An error occurred during training: {e}")
            self.update_status("Training failed.")
            self.master.after(0, lambda: self.set_buttons_state(tk.DISABLED))

    def start_prediction_thread(self):
        """Starts model prediction in a separate thread."""
        self.update_status("Loading model and making predictions...")
        self.set_buttons_state(tk.DISABLED)

        prediction_thread = threading.Thread(target=self._predict_task)
        prediction_thread.daemon = True
        prediction_thread.start()

    def _predict_task(self):
        """Task for prediction thread. Calls the model and updates GUI with results."""
        try:
            results = self.stock_model.predict(self.train_df, self.test_df)
            self.y_test_original = results["y_test_original"]
            self.y_pred_original = results["y_pred_original"]
            self.metrics_data = results["metrics"]

            # Display evaluation metrics
            self.update_output("\n--- Evaluation Metrics On Testing ---")
            for name, value in self.metrics_data.items():
                self.update_output(f"📊 {name.replace('_', ' '):<8}: {value:.3f}")
            # --- Display the entire testing dataset with robust column check ---
            self.update_output("\n--- Testing Data (Date and Close Price) ---")
            try:
                required_columns = ['Date', 'Close']
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
           
            self.update_status("Prediction complete. Results displayed.")
            # Enable buttons after prediction
            self.master.after(0, lambda: self.set_buttons_state(tk.NORMAL))

        except Exception as e:
            self.update_output(f"An error occurred during prediction: {e}")
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.update_status("Prediction failed.")
            self.master.after(0, lambda: self.set_buttons_state(tk.DISABLED))


    def start_future_prediction_thread(self):
        """Starts 10-day future prediction in a separate thread."""
        self.update_status("Predicting next 10 days...")
        self.update_output(f"\n--- Prediction of the Next 10 Days After {self.test_filename_for_plot} ---")
        self.predict_10_days_button.config(state=tk.DISABLED)

        prediction_thread = threading.Thread(target=self._predict_future_task)
        prediction_thread.daemon = True
        prediction_thread.start()

    def _predict_future_task(self):
        """Task for future prediction thread."""
        try:
            future_predictions = self.stock_model.predict_future(self.train_df, self.test_df, 10)
            for i, prediction in enumerate(future_predictions):
                self.update_output(f"Predicted Day {i + 1}: {prediction:.2f}")
                time.sleep(0.1)
            self.update_status("10-day future prediction complete.")
        except Exception as e:
            self.update_output(f"An error occurred during future prediction: {e}")
            messagebox.showerror("Prediction Error", f"An error occurred during future prediction: {e}")
            self.update_status("10-day future prediction failed.")
        finally:
            # Re-enable the button
            self.master.after(0, lambda: self.predict_10_days_button.config(state=tk.NORMAL))

    def set_buttons_state(self, state):
        """Enable or disable plot and prediction buttons based on available data."""
        self.loss_plot_button.config(state=state if self.history else tk.DISABLED)
        self.prediction_plot_button.config(state=state if self.y_pred_original is not None else tk.DISABLED)
        self.metrics_plot_button.config(state=state if self.metrics_data else tk.DISABLED)
        self.predict_10_days_button.config(state=state if self.test_df is not None else tk.DISABLED)

    def show_loss_plot(self):
        """Displays the model's training loss plot in a new window."""
        if self.history is None:
            messagebox.showwarning("No Data", "Model has not been trained yet.")
            return
        self.create_plot_window(f'Model Loss During Training on({self.train_filename_for_plot})', self.plot_loss)

    def show_prediction_plot(self):
        """Displays the actual vs. predicted prices plot in a new window."""
        if self.y_test_original is None:
            messagebox.showwarning("No Data", "Predictions have not been made yet.")
            return
        self.create_plot_window(f"Prediction for {self.test_filename_for_plot}", self.plot_predictions)

    def show_evaluation_metrics_plot(self):
        """Displays a bar chart of the evaluation metrics in a new window."""
        if not self.metrics_data:
            messagebox.showwarning("No Data", "Evaluation metrics have not been calculated yet.")
            return
        self.create_plot_window("Evaluation Metrics Overview", self.plot_metrics)
        
    def create_plot_window(self, title, plot_function):
        """Generic function to create a Toplevel window for matplotlib plots."""
        plot_window = tk.Toplevel(self.master)
        plot_window.title(title)
        plot_window.geometry("800x600")

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_function(ax) # Call the specific plot function to draw on the axes
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a toolbar for zooming, panning, etc.
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def plot_loss(self, ax):
        """Helper to draw the loss plot on given axes."""
        ax.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_title(f'Model Loss During Training on {self.train_filename_for_plot}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
    def plot_predictions(self, ax):
        """Helper to draw the prediction plot on given axes."""
        ax.plot(self.y_test_original, label="Actual Prices", color='blue', linewidth=2)
        ax.plot(self.y_pred_original, label="Predicted Prices", color='orange', linestyle='--', linewidth=2)
        ax.set_title(f"LSTM Stock Price Prediction ({self.test_filename_for_plot})")
        ax.set_xlabel(f"Time Step (Days in Test Set)")
        ax.set_ylabel("Stock Close Price")
        ax.legend()
        ax.grid(True)

    def plot_metrics(self, ax):
        """Helper to draw the metrics bar chart on given axes."""
        names = list(self.metrics_data.keys())
        values = list(self.metrics_data.values())
        colors = ['skyblue', 'lightcoral', 'silver', 'gold']
        bars = ax.bar(names, values, color=colors)
        ax.set_title(f"Model Evaluation Metrics on {self.test_filename_for_plot}")
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        # Add value labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    def clear_output(self):
        """Clears the output text area and resets the application state."""
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        self.update_status("Output cleared.")
        
        # Reset all relevant data
        self.y_test_original = None
        self.y_pred_original = None
        self.history = None
        self.metrics_data = {}
        self.train_df = None
        self.test_df = None
        self.train_file_path.set("")
        self.test_file_path.set("")
        
        self.check_saved_model_status()

    def check_saved_model_status(self):
        """Checks if a saved model exists and updates GUI accordingly."""
        if self.stock_model.check_model_exists():
            self.update_status("Ready to load data and predict.")
        else:
            self.update_status("No saved model/scaler found. Please load training data to begin.")
        self.set_buttons_state(tk.DISABLED)

    def back_to_welcome(self):
        """Hides the main app and calls the callback to show the welcome page."""
        if self.on_back_callback:
            self.master.withdraw()
            self.on_back_callback()
        else:
            messagebox.showwarning("Navigation Error", "No callback function provided to go back.")
        
    def end_application(self):
        """Confirms with the user and then terminates the application."""
        if messagebox.askyesno("Exit Application", "Are you sure you want to exit?"):
            self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()

