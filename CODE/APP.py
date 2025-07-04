# Code for creating a flask server with our own certificates to work like a website
import os
import socket
import qrcode
import base64
import platform
import webbrowser
import numpy as np
import joblib
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image

# --- CONFIGURATION ---
TEMPLATE_DIR = r"C:\Face Recognition Dataset\My project\template"
MODEL_DIR = r"C:\Face Recognition Dataset\My project\models\svm"
ATTENDANCE_FILE = "attendance.csv"
CERT_FILE = r"C:\Face Recognition Dataset\My project\template\cert.pem"
KEY_FILE = r"C:\Face Recognition Dataset\My project\template\key.pem"

# --- FLASK INIT ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- SERVER URL SETUP ---
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
server_url = f"https://{local_ip}:5000"  # HTTPS

# --- ATTENDANCE TRACKER ---
marked_attendance = set()

# --- Load Pre-trained Models ---
def load_model_assets():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "svm_final_tuned_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
        label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.pkl"))
        return model, scaler, pca, label_map
    except Exception as e:
        print(f"[❌] Model loading error: {e}")
        raise

model, scaler, pca, label_map = load_model_assets()

# --- QR Code Generator ---
def generate_qr_code():
    try:
        qr = qrcode.make(server_url)
        qr_path = os.path.join(os.getcwd(), "qr_code.png")
        qr.save(qr_path)
        print(f"[✅] QR saved: {qr_path}")
        print(f"[📱] Scan to access: {server_url}")
        if platform.system() == "Windows":
            os.startfile(qr_path)
        else:
            os.system(f"open {qr_path}")
    except Exception as e:
        print(f"[❌] QR Generation Failed: {e}")

# --- Feature Extractors ---
def compute_lbp_hist(image, radius=3, n_points=24):
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, int(lbp.max()) + 2), density=True)
    return hist

def compute_hog_features(image):
    from skimage.feature import hog
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

# --- ROUTES ---
@app.route('/')
def index():
    try:
        return render_template("face.htm")
    except Exception as e:
        return f"<h2>Server Error</h2><p>{e}</p>", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Invalid data received'}), 400

        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('L').resize((100, 100))
        img_np = np.array(img)

        lbp_hist = compute_lbp_hist(img_np)
        hog_feat = compute_hog_features(img_np)
        features = np.hstack((lbp_hist, hog_feat))
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)

        prediction = model.predict(features_pca)[0]
        label = label_map[prediction]

        if label not in marked_attendance:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(ATTENDANCE_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label, now])
            marked_attendance.add(label)
            print(f"[📋] Attendance marked for {label} at {now}")

        return jsonify({'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- RUN SERVER ---
if __name__ == '__main__':
    print("[🚀] Starting Flask Server with HTTPS...")
    generate_qr_code()
    print(f"[🌐] Server running at: {server_url}")
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=(CERT_FILE, KEY_FILE))
