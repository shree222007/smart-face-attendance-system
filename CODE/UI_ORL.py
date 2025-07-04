# This is the Code for a simple UI to upload files 
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import joblib
import os
import csv
from datetime import datetime

# Load model and PCA
svm = joblib.load(r"C:\Face Recognition Dataset\olivetti dataset\svm_model.pkl")
pca = joblib.load(r"C:\Face Recognition Dataset\olivetti dataset\pca_model.pkl")

# CSV attendance file
csv_file = r"C:\Face Recognition Dataset\My project\template\prediction_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Predicted ID", "Date", "Time"])

def get_today_marked_ids():
    marked = set()
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Date'] == today:
                    marked.add(row['Predicted ID'])
    return marked

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not file_path:
        return

    img = Image.open(file_path).convert('L').resize((64, 64))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_flat = img_np.flatten().reshape(1, -1)
    img_pca = pca.transform(img_flat)
    pred = svm.predict(img_pca)[0]
    pred_str = str(pred)

    # Show image
    img_disp = ImageTk.PhotoImage(img.resize((200, 200)))
    label_img.configure(image=img_disp)
    label_img.image = img_disp

    # Show prediction
    label_result.config(text=f"🎯 Predicted Person ID: {pred_str}")

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if pred_str not in get_today_marked_ids():
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([pred_str, date_str, time_str])
        print(f"[✅] Attendance marked for ID {pred_str}")
    else:
        print(f"[ℹ️] ID {pred_str} already marked today.")

# --- GUI SETUP ---

root = tk.Tk()
root.title("Face Recognition - Olivetti PCA + SVM")
root.geometry("600x600")
root.configure(bg="black")

# Heading
title = tk.Label(root, text="Face Recognition System", font=("Segoe UI", 22, "bold"), bg="black", fg="white")
title.grid(row=0, column=0, columnspan=2, pady=(20, 5))

subtitle = tk.Label(root, text="Upload a face image to identify the person", font=("Segoe UI", 12), bg="black", fg="white")
subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 10))

# Image preview area
preview_frame = tk.Frame(root, width=300, height=300, bg="gray20", bd=2, relief="sunken")
preview_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10)
preview_frame.grid_propagate(False)

label_img = tk.Label(preview_frame, bg="gray20")
label_img.pack(expand=True)

# Upload Button
style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12), padding=6)
upload_btn = ttk.Button(root, text="📤 Upload Image", command=predict_image)
upload_btn.grid(row=3, column=0, columnspan=2, pady=15)

# Prediction result (centered)
label_result = tk.Label(root, text="", font=("Segoe UI", 14), fg="white", bg="black")
label_result.grid(row=4, column=0, columnspan=2, pady=5, sticky="nsew")

# Ensure the content stretches and centers
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()
