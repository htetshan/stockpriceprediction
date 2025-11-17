import tkinter as tk
from tkinter import font, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import sys

# --- UPDATED IMPORT ---
# Import the main GUI application from the new 'main_gui.py' file
from main_gui import StockPredictorApp

class WelcomePage(tk.Toplevel):
    """
    Creates and manages the welcome screen GUI.
    """
    def __init__(self, master, on_start_callback):
        super().__init__(master)
        self.master = master
        self.on_start_callback = on_start_callback
        
        self.title("Welcome")
        
        initial_width = 1200
        initial_height = 700
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (initial_width // 2)
        y = (screen_height // 2) - (initial_height // 2)
        self.geometry(f"{initial_width}x{initial_height}+{x}+{y}")
        self.resizable(True, True)

        self.background_color = "#BCC3A6" 
        self.configure(bg=self.background_color) 
        
        self.title_font = font.Font(family="Times New Roman", size=24)
        self.subtitle_font = font.Font(family="Times New Roman", size=24)
        self.main_text_font = font.Font(family="Times New Roman", size=30, weight="bold")
        self.mid_text_font = font.Font(family="Times New Roman", size=18)
        self.footer_label_font = font.Font(family="Times New Roman", size=18)
        
        self.create_widgets()

    def start_application(self):
        """Hides the welcome page and calls the callback to start the main app."""
        self.withdraw()
        self.on_start_callback()

    def create_widgets(self):
        """Builds all the GUI elements for the welcome screen."""
        top_frame = tk.Frame(self, bg=self.background_color, pady=20) 
        top_frame.pack(fill="x")
        
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=6)
        top_frame.grid_columnconfigure(2, weight=1)
        
        try:
            logo_left_path = "assets/logo_left.png"  
            logo_left_img = Image.open(logo_left_path).resize((130, 130), Image.Resampling.LANCZOS)
            self.logo_left_tk = ImageTk.PhotoImage(logo_left_img)
            tk.Label(top_frame, image=self.logo_left_tk, bg=self.background_color).grid(row=0, column=0, rowspan=2, padx=20, pady=10, sticky="w")
        except FileNotFoundError:
            tk.Label(top_frame, text="Left Logo", font=self.footer_label_font, bg=self.background_color).grid(row=0, column=0, rowspan=2)

        tk.Label(top_frame, text="TECHNOLOGICAL UNIVERSITY (MEIKTILA)", font=self.title_font, bg=self.background_color).grid(row=0, column=1, pady=(10, 0))
        tk.Label(top_frame, text="DEPARTMENT OF INFORMATION TECHNOLOGY", font=self.subtitle_font, bg=self.background_color).grid(row=1, column=1, pady=(0, 10))

        try:
            logo_right_path = "assets/logo_right.png"
            logo_right_img = Image.open(logo_right_path).resize((130, 130), Image.Resampling.LANCZOS)
            self.logo_right_tk = ImageTk.PhotoImage(logo_right_img)
            tk.Label(top_frame, image=self.logo_right_tk, bg=self.background_color).grid(row=0, column=2, rowspan=2, padx=20, pady=10, sticky="e")
        except FileNotFoundError:
            tk.Label(top_frame, text="Right Logo", font=self.footer_label_font, bg=self.background_color).grid(row=0, column=2, rowspan=2)

        middle_frame = tk.Frame(self, bg=self.background_color) 
        middle_frame.pack(fill="x", pady=60)
        tk.Label(middle_frame, text="Stock Price Prediction Using Long Short-Term Memory", font=self.main_text_font, justify="center", bg=self.background_color).pack(pady=20)
        
        tk.Frame(self, bg=self.background_color).pack(fill="both", expand=True)

        bottom_frame = tk.Frame(self, bg=self.background_color, pady=40) 
        bottom_frame.pack(fill="x")
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)
        
        supervised_frame = tk.Frame(bottom_frame, bg=self.background_color)
        supervised_frame.grid(row=0, column=0, padx=50, sticky="sw")
        tk.Label(supervised_frame, text="SUPERVISED BY:", font=self.footer_label_font, bg=self.background_color).pack(anchor="w")
        tk.Label(supervised_frame, text="Dr. Nu Nu Htway", font=self.footer_label_font, bg=self.background_color, justify="center").pack(anchor="w")
        
        tk.Button(bottom_frame, text="Start", command=self.start_application, font=("Arial", 16, "bold"), bg="#4CAF50", fg="white", relief=tk.RAISED, bd=4, padx=30, pady=10).grid(row=0, column=1, padx=20, pady=20)

        presented_frame = tk.Frame(bottom_frame, bg=self.background_color)
        presented_frame.grid(row=0, column=2, padx=50, sticky="se")
        tk.Label(presented_frame, text="PRESENTED BY:", font=self.footer_label_font, bg=self.background_color).pack(anchor="e")
        tk.Label(presented_frame, text="Mg Htet Shan Kyaw\n(VI-IT-9)", font=self.footer_label_font, justify="center", bg=self.background_color).pack(anchor="e")

# Main execution block
if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")

    root = tk.Tk()
    root.withdraw()

    app_instance = None
    welcome_page_instance = None
    
    def show_welcome_page_callback():
        """Hides the main app and shows the welcome page."""
        root.withdraw()
        if welcome_page_instance and welcome_page_instance.winfo_exists():
            welcome_page_instance.deiconify()
        else:
            print("Warning: WelcomePage instance was destroyed.")

    def start_main_app_callback():
        """Starts the main StockPredictorApp."""
        global app_instance
        root.deiconify() 
        if app_instance is None:
            try:
                app_instance = StockPredictorApp(root, on_back_callback=show_welcome_page_callback)
            except Exception as e:
                messagebox.showerror("Application Error", f"Failed to start main application: {e}\nCheck console for details.")
                root.destroy()
    
    try:
        welcome_page_instance = WelcomePage(root, start_main_app_callback)
        welcome_page_instance.deiconify() 
        root.mainloop() 
    except Exception as e:
        messagebox.showerror("Startup Error", f"An error prevented the application from starting: {e}\nCheck console for details.")
        root.destroy()

    print("--- Application finished ---")
