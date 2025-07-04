# Face Recognition Attendance System (Flask + HTTPS)

A Flask-based HTTPS face recognition attendance system using PCA, LBP, HOG, and SVM with QR code access. This project is ideal for secure, real-time attendance marking using facial recognition over a local network.

---

##  Features

-  **Face Recognition** using PCA + LBP + HOG features
-  **SVM Classifier** fine-tuned with GridSearchCV
-  **Secure HTTPS** Flask server with your own SSL certificate
-  **Mobile Access via QR Code**
-  **Attendance Logging** with timestamps in a CSV file

---

##  Folder Structure

```
face-recognition-flask-attendance/
├── main.py
├── attendance.csv
├── models/
│   ├── svm_final_tuned_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── label_map.pkl
├── template/
│   ├── face.htm
│   ├── cert.pem
│   └── key.pem
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  How It Works

- Accepts a webcam-captured image (base64) from the front-end
- Extracts features using LBP and HOG
- Reduces dimensions using PCA
- Classifies using the trained SVM model
- Marks attendance in `attendance.csv` if not already marked

---

##  Requirements

- Python 3.8+
- Flask
- OpenCV
- Scikit-learn
- Scikit-image
- NumPy, Matplotlib, Seaborn
- Pillow, joblib, qrcode

---

## 🛡️ License

This project is licensed under the MIT License.
