import tkinter as tk
from tkinter import font, messagebox # Import messagebox explicitly
from PIL import Image, ImageTk
import tensorflow as tf
import sys # Import sys for better error handling

# Ensure this import correctly points to your local file
# The StockPredictorApp class is assumed to be defined in stock_predictor_gui.py
from stock_predictor_gui import StockPredictorApp

class WelcomePage(tk.Toplevel):
    """
    Creates and manages the welcome screen GUI based on the corrected design.
    It now accepts an on_start_callback for integration with the main application.
    """
    def __init__(self, master, on_start_callback):
        super().__init__(master) # Initialize Toplevel
        self.master = master
        self.on_start_callback = on_start_callback # Store the callback
        
        self.title("Welcome To My Project")
        
        # Set initial geometry to a 3:2 width:height ratio, e.g., 1050x700
        initial_width = 1050
        initial_height = 700
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (initial_width // 2)
        y = (screen_height // 2) - (initial_height // 2)
        self.geometry(f"{initial_width}x{initial_height}+{x}+{y}")
        self.resizable(True, True) # Allow resizing the window and use OS's maximize button

        # Set a pure white background color
        self.background_color = "#ffffff" 
        self.configure(bg=self.background_color) 
        
        # Create and configure fonts for consistent styling
        # Reduced font sizes slightly to help with text fitting on screen
        self.title_font = font.Font(family="Times New Roman", size=20,) # Reduced from 30
        self.subtitle_font = font.Font(family="Times New Roman", size=20,) # Reduced from 18
        self.main_text_font = font.Font(family="Times New Roman", size=24, weight="bold")
        self.mid_text_font = font.Font(family="Times New Roman", size=18)
        self.footer_label_font = font.Font(family="Times New Roman", size=14)
        
        self.create_widgets()

    def start_application(self):
        """
        Hides the welcome page and calls the stored callback to start the main app.
        Changed from destroy() to withdraw() to allow returning to this page.
        """
        print("Start button clicked. Hiding welcome screen and triggering main app.")
        self.withdraw() # Hide this Toplevel window instead of destroying it
        self.on_start_callback() # Call the function passed from main_app.py

    def create_widgets(self):
        """Builds all the GUI elements for the welcome screen."""
        
        # --- Top Section ---
        top_frame = tk.Frame(self, bg=self.background_color, pady=20) 
        top_frame.pack(fill="x")
        
        # Adjusted column weights to give more space to the center text
        # Changed float weights to integers as required by grid_columnconfigure
        top_frame.grid_columnconfigure(0, weight=1) # Less weight for left logo
        top_frame.grid_columnconfigure(1, weight=6)   # More weight for center text (3 / 0.5 = 6)
        top_frame.grid_columnconfigure(2, weight=1) # Less weight for right logo
        
        # Left Logo
        try:
            logo_left_path = "assets/logo_left.png"  
            logo_left_img = Image.open(logo_left_path)
            logo_left_img = logo_left_img.resize((150, 150), Image.Resampling.LANCZOS)
            self.logo_left_tk = ImageTk.PhotoImage(logo_left_img)
            
            logo_left_label = tk.Label(top_frame, image=self.logo_left_tk, bg=self.background_color)
            logo_left_label.grid(row=0, column=0, rowspan=2, padx=20, pady=10, sticky="w")
        except FileNotFoundError:
            print(f"Error: Logo file not found at {logo_left_path}")
            tk.Label(top_frame, text="Left Logo", font=self.footer_label_font, bg=self.background_color).grid(row=0, column=0, rowspan=2)

        # Center Text
        university_label = tk.Label(top_frame, 
                                    text="TECHNOLOGICAL UNIVERSITY (MEIKTILA)", 
                                    font=self.title_font, 
                                    bg=self.background_color)
        university_label.grid(row=0, column=1, pady=(10, 0))
        
        department_label = tk.Label(top_frame, 
                                    text="DEPARTMENT OF INFORMATION TECHNOLOGY", 
                                    font=self.subtitle_font, 
                                    bg=self.background_color)
        department_label.grid(row=1, column=1, pady=(0, 10))

        # Right Logo
        try:
            logo_right_path = "assets/logo_right.png"  
            logo_right_img = Image.open(logo_right_path)
            logo_right_img = logo_right_img.resize((150, 150), Image.Resampling.LANCZOS)
            self.logo_right_tk = ImageTk.PhotoImage(logo_right_img)
            
            logo_right_label = tk.Label(top_frame, image=self.logo_right_tk, bg=self.background_color)
            logo_right_label.grid(row=0, column=2, rowspan=2, padx=20, pady=10, sticky="e")
        except FileNotFoundError:
            print(f"Error: Logo file not found at {logo_right_path}")
            tk.Label(top_frame, text="Right Logo", font=self.footer_label_font, bg=self.background_color).grid(row=0, column=2, rowspan=2)

        # --- Middle Section ---
        middle_frame = tk.Frame(self, bg=self.background_color) 
        middle_frame.pack(fill="x", pady=50)

        project_title_label = tk.Label(middle_frame, 
                                      text="Stock Price Prediction Using Long Short-Term Memory(LSTM)",
                                      font=self.main_text_font,
                                      justify="center",
                                      bg=self.background_color)
        project_title_label.pack(pady=20)
        
        seminar_label = tk.Label(middle_frame, 
                                 text="",
                                 font=self.mid_text_font,
                                 justify="center",
                                 bg=self.background_color)
        seminar_label.pack(pady=(20, 0))
        
        date_label = tk.Label(middle_frame, 
                              text="",
                              font=self.mid_text_font,
                              justify="center",
                              bg=self.background_color)
        date_label.pack(pady=(0, 20))
        
        # --- Spacer to push the bottom elements down ---
        spacer_frame = tk.Frame(self, bg=self.background_color) 
        spacer_frame.pack(fill="both", expand=True)

        # --- Bottom Section ---
        bottom_frame = tk.Frame(self, bg=self.background_color, pady=40) 
        bottom_frame.pack(fill="x")
        
        # Changed float weights to integers as required by grid_columnconfigure
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)
        
        # Supervised By (Left column)
        supervised_frame = tk.Frame(bottom_frame, bg=self.background_color)
        supervised_frame.grid(row=0, column=0, padx=50, sticky="sw")
        
        supervised_label_1 = tk.Label(supervised_frame, text="SUPERVISED BY:", font=self.footer_label_font, bg=self.background_color)
        supervised_label_1.pack(anchor="w")
        
        supervised_label_2 = tk.Label(supervised_frame, text="Dr. Nu Nu Htway", font=self.footer_label_font, bg=self.background_color,justify="center")
        supervised_label_2.pack(anchor="w")
        
        # Start Button (Center column)
        start_button = tk.Button(bottom_frame, 
                                 text="Start",
                                 command=self.start_application, 
                                 font=("Arial", 16, "bold"),
                                 bg="#4CAF50",
                                 fg="white",
                                 relief=tk.RAISED,
                                 bd=4,
                                 padx=30,
                                 pady=10)
        start_button.grid(row=0, column=1, padx=20, pady=20)

        # Presented By (Right column)
        presented_frame = tk.Frame(bottom_frame, bg=self.background_color)
        presented_frame.grid(row=0, column=2, padx=50, sticky="se")
        
        presented_label_1 = tk.Label(presented_frame, text="PRESENTED BY:", font=self.footer_label_font, bg=self.background_color)
        presented_label_1.pack(anchor="e")
        
        presented_label_2 = tk.Label(presented_frame, text="Mg Htet Shan Kyaw\n(VI-IT-9)", font=self.footer_label_font, justify="center", bg=self.background_color)
        presented_label_2.pack(anchor="e")


# Main part of the script
if __name__ == "__main__":
    print("--- Starting combined app.py ---")

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

    # Keep references to app and welcome_page instances outside the function scope
    app_instance = None
    welcome_page_instance = None
    
    def show_welcome_page_callback():
        """
        Callback function to hide the main app and show the welcome page.
        """
        print("show_welcome_page_callback triggered.")
        root.withdraw() # Hide the main app window (root)
        if welcome_page_instance: # Check if welcome_page instance exists and is not destroyed
            if welcome_page_instance.winfo_exists(): # Check if the Toplevel window still exists
                welcome_page_instance.deiconify() # Show the welcome page
                print("Returned to WelcomePage.")
            else:
                print("Warning: WelcomePage instance was destroyed. Cannot deiconify.")
        else:
            print("Warning: WelcomePage instance not found for 'Back' functionality.")

    def start_main_app_callback():
        """
        Callback function to deiconify the main window and start the StockPredictorApp.
        This function is passed to the WelcomePage.
        """
        global app_instance # Declare intent to modify global variable
        print("start_main_app_callback triggered.")
        
        # Ensure the root window is deiconified before creating/showing the app
        root.deiconify() 
        print("Main window deiconified.")

        if app_instance is None: # Create app only once
            try:
                # Pass the show_welcome_page_callback to StockPredictorApp
                app_instance = StockPredictorApp(root, on_back_callback=show_welcome_page_callback)
                print("StockPredictorApp initialized successfully with back callback.")
            except Exception as e:
                print(f"ERROR: Failed to initialize StockPredictorApp: {e}", file=sys.stderr)
                messagebox.showerror("Application Error", f"Failed to start main application: {e}\nCheck console for details.")
                root.destroy() # Close the root window if main app fails to start
        else:
            print("StockPredictorApp already initialized, showing its content.")

    try:
        # Create the WelcomePage instance and store it globally
        welcome_page_instance = WelcomePage(root, start_main_app_callback)
        # Explicitly show the welcome page as root is initially withdrawn.
        # This is important because Toplevel windows are not automatically shown
        # if their master is withdrawn.
        welcome_page_instance.deiconify() 
        print("WelcomePage initialized and shown. Entering mainloop...")
        
        # The mainloop will keep both the root and the Toplevel (welcome_page) alive
        root.mainloop() 
        print("Exited mainloop.")
    except Exception as e:
        print(f"ERROR: An error occurred during WelcomePage initialization or mainloop: {e}", file=sys.stderr)
        messagebox.showerror("Startup Error", f"An error prevented the application from starting: {e}\nCheck console for details.")
        root.destroy() # Ensure root window is destroyed on critical startup error

    print("--- Application finished ---")
