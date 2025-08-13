import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

class WelcomeScreen:
    """
    Creates and manages the welcome screen GUI based on the corrected design.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("Project Welcome Screen")
        self.master.geometry("1000x700")
        self.master.resizable(False, False)
        
        # Set a light, off-white background color
        self.background_color = "#f5f5dc"
        self.master.configure(bg=self.background_color)
        
        # Create and configure fonts for consistent styling
        self.title_font = font.Font(family="Times New Roman", size=30, weight="bold")
        self.subtitle_font = font.Font(family="Times New Roman", size=18)
        self.main_text_font = font.Font(family="Times New Roman", size=24, weight="bold")
        self.mid_text_font = font.Font(family="Times New Roman", size=18)
        self.footer_label_font = font.Font(family="Times New Roman", size=14)
        
        self.create_widgets()

    def start_application(self):
        """
        This is the function that will be called when the Start button is clicked.
        You can put the code to open your main application here.
        For this example, it will simply close the welcome screen.
        """
        print("Start button clicked. Starting the main application...")
        self.master.destroy()

    def create_widgets(self):
        """Builds all the GUI elements for the welcome screen."""
        
        # --- Top Section ---
        top_frame = tk.Frame(self.master, bg=self.background_color, pady=20)
        top_frame.pack(fill="x")
        
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=2)
        top_frame.grid_columnconfigure(2, weight=1)
        
        # Left Logo
        try:
            # Replace 'logo_left.png' with your actual image path
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
            # Replace 'logo_right.png' with your actual image path
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
        middle_frame = tk.Frame(self.master, bg=self.background_color)
        middle_frame.pack(fill="x", pady=50)

        project_title_label = tk.Label(middle_frame, 
                                      text="Stock Price Prediction Using Long Short-Term Memory(LSTM)",
                                      font=self.main_text_font,
                                      justify="center",
                                      bg=self.background_color)
        project_title_label.pack(pady=20)
        
        seminar_label = tk.Label(middle_frame, 
                                 text="Second Seminar",
                                 font=self.mid_text_font,
                                 justify="center",
                                 bg=self.background_color)
        seminar_label.pack(pady=(20, 0))
        
        date_label = tk.Label(middle_frame, 
                              text="28.2.2025",
                              font=self.mid_text_font,
                              justify="center",
                              bg=self.background_color)
        date_label.pack(pady=(0, 20))
        
        # --- Spacer to push the bottom elements down ---
        spacer_frame = tk.Frame(self.master, bg=self.background_color)
        spacer_frame.pack(fill="both", expand=True)

        # --- Bottom Section ---
        bottom_frame = tk.Frame(self.master, bg=self.background_color, pady=40)
        bottom_frame.pack(fill="x")
        
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)
        
        # Supervised By (Left column)
        supervised_frame = tk.Frame(bottom_frame, bg=self.background_color)
        supervised_frame.grid(row=0, column=0, padx=50, sticky="sw")
        
        supervised_label_1 = tk.Label(supervised_frame, text="SUPERVISED BY:", font=self.footer_label_font, bg=self.background_color)
        supervised_label_1.pack(anchor="w")
        
        supervised_label_2 = tk.Label(supervised_frame, text="DAW NU NU HTWAY", font=self.footer_label_font, bg=self.background_color)
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
        
        presented_label_2 = tk.Label(presented_frame, text="MG HTET SHAN KYAW\n(VI-IT-9)", font=self.footer_label_font, justify="right", bg=self.background_color)
        presented_label_2.pack(anchor="e")


if __name__ == "__main__":
    root = tk.Tk()
    app = WelcomeScreen(root)
    root.mainloop()