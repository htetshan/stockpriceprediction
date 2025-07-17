import tkinter as tk

class WelcomePage(tk.Toplevel):
    """
    A simple welcome page that appears before the main application.
    """
    def __init__(self, master, on_start_callback):
        super().__init__(master)
        self.master = master
        self.on_start_callback = on_start_callback

        self.title("Welcome") # Simpler title
        # Significantly increased window size to better fit your screen
        self.geometry("700x450") # Changed from 550x350 to 700x450
        self.resizable(False, False)
        
        # Center the welcome window
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (self.winfo_width() // 2)
        y = (screen_height // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

        # Removed grab_set() and transient() as they caused issues previously
        # self.grab_set()
        # self.transient(master)

        try:
            self.create_widgets()
        except Exception as e:
            # messagebox imported in main_app.py, so we can't use it directly here
            # print(f"ERROR: Failed to create WelcomePage widgets: {e}") # Keep for console debugging if needed
            self.destroy() # Destroy welcome page if widget creation fails

    def create_widgets(self):
        """Creates the widgets for the simple welcome page."""
        welcome_label = tk.Label(self, 
                                 text="Welcome!", # Simpler welcome text
                                 font=("Arial", 36, "bold"), # Much larger font for prominence
                                 fg="#1E88E5", # A pleasant blue color
                                 pady=50) # Increased padding for spacing
        welcome_label.pack()

        info_text = "Click below to start the Stock Price Prediction System." # Simpler info text
        info_label = tk.Label(self, 
                              text=info_text,
                              font=("Arial", 16), # Larger font
                              justify=tk.CENTER,
                              wraplength=600, # Adjusted wraplength to fit new window width
                              pady=30) # Increased padding
        info_label.pack(padx=20, pady=10)

        start_button = tk.Button(self, 
                                 text="Start", # Simple button text
                                 command=self.start_application,
                                 bg="#4CAF50", # Green background
                                 fg="white", # White text
                                 font=("Arial", 18, "bold"), # Larger, bold font
                                 relief=tk.RAISED,
                                 bd=4, # Slightly thicker border
                                 padx=40, # Adjusted padding
                                 pady=20) # Adjusted padding
        start_button.pack(pady=40) # Adjusted padding

    def start_application(self):
        """Destroys the welcome page and calls the callback to start the main app."""
        self.destroy()
        self.on_start_callback()
