
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
from lstm_training_module import train_lstm_model

class LSTMStockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 LSTM Stock Prediction App")
        self.train_csv_path = None
        self.test_csv_path = None
        self.history = None

        # Buttons
        tk.Button(root, text="📂 Load Train CSV", width=30, command=self.load_train_csv).pack(pady=5)
        tk.Button(root, text="📂 Load Test CSV", width=30, command=self.load_test_csv).pack(pady=5)
        tk.Button(root, text="🧠 Train Model", width=30, command=self.train_model).pack(pady=5)
        tk.Button(root, text="📈 Show Loss Figure", width=30, command=self.show_loss_plot).pack(pady=5)
        tk.Button(root, text="🧹 Clear", width=30, command=self.clear_output).pack(pady=5)

        # Output box
        self.output_box = scrolledtext.ScrolledText(root, width=60, height=20)
        self.output_box.pack(padx=10, pady=10)

    def load_train_csv(self):
        path = filedialog.askopenfilename(title="Select Training CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.train_csv_path = path
            self.output_box.insert(tk.END, f"✅ Loaded Train CSV: {path}\n")

    def load_test_csv(self):
        path = filedialog.askopenfilename(title="Select Testing CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.test_csv_path = path
            self.output_box.insert(tk.END, f"✅ Loaded Test CSV: {path}\n")

    def train_model(self):
        if not self.train_csv_path:
            messagebox.showwarning("Missing File", "Please load training CSV first.")
            return

        try:
            self.output_box.insert(tk.END, "🧠 Training model...\n")
            self.output_box.update()

            model_path = "saved_lstm_model.h5"
            scaler_path = "saved_scaler.pkl"

            model, history = train_lstm_model(
                train_csv_path=self.train_csv_path,
                model_save_path=model_path,
                scaler_save_path=scaler_path
            )

            self.output_box.insert(tk.END, f"✅ Model trained and saved to: {model_path}\n")
            self.output_box.insert(tk.END, f"✅ Scaler saved to: {scaler_path}\n")

            self.history = history

        except Exception as e:
            self.output_box.insert(tk.END, f"❌ Error during training: {str(e)}\n")

    def show_loss_plot(self):
        try:
            import matplotlib.pyplot as plt
            if self.history is None:
                self.output_box.insert(tk.END, "⚠️ No training history found. Train the model first.\n")
                return
            plt.figure(figsize=(10, 5))
            plt.plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.output_box.insert(tk.END, f"❌ Error showing plot: {str(e)}\n")

    def clear_output(self):
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, "🧹 Output cleared.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = LSTMStockApp(root)
    root.mainloop()
