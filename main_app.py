import tkinter as tk
import tensorflow as tf
# Ensure these imports correctly point to your local files
from welcome_page import WelcomePage
from stock_predictor_gui import StockPredictorApp
import sys # Import sys for better error handling

# Main part of the script
if __name__ == "__main__":
    print("--- Starting main_app.py ---")

    # It's good practice to ensure TensorFlow doesn't hog all GPU memory if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected.")
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")
            # This error might happen if TensorFlow can't find a GPU or has issues with it.
            # The app should still run on CPU, but it's good to know.

    root = tk.Tk()
    print("Tkinter root window created.")
    root.withdraw() # Hide the main window initially
    print("Main window hidden (root.withdraw()).")

    def start_main_app():
        """
        Callback function to deiconify the main window and start the StockPredictorApp.
        """
        print("start_main_app callback triggered.")
        root.deiconify() # Show the main window
        print("Main window deiconified.")
        try:
            app = StockPredictorApp(root)
            print("StockPredictorApp initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize StockPredictorApp: {e}", file=sys.stderr)
            messagebox.showerror("Application Error", f"Failed to start main application: {e}\nCheck console for details.")
            root.destroy() # Close the root window if main app fails to start
        # Note: root.mainloop() is already running due to the WelcomePage,
        # so no need to call it again here.

    try:
        welcome_page = WelcomePage(root, start_main_app)
        print("WelcomePage initialized. Entering mainloop...")
        # The mainloop will keep both the root and the Toplevel (welcome_page) alive
        root.mainloop() 
        print("Exited mainloop.")
    except Exception as e:
        print(f"ERROR: An error occurred during WelcomePage initialization or mainloop: {e}", file=sys.stderr)
        messagebox.showerror("Startup Error", f"An error prevented the application from starting: {e}\nCheck console for details.")
        root.destroy() # Ensure root window is destroyed on critical startup error

    print("--- Application finished ---")

