import tkinter as tk

print("Creating simple Tkinter window...")
root = tk.Tk()
root.title("Simple Test Window")
root.geometry("300x200")
tk.Label(root, text="If you see this, Tkinter is working!").pack(pady=20)
print("Entering mainloop for simple window...")
root.mainloop()
print("Exited simple window mainloop.")