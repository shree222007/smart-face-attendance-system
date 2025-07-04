# This is the model code for FNN for ORL dataset
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Dataset Directory ---
dataset_dir = r"C:\Face Recognition Dataset\My project\preprocessed"

# --- Load Images and Labels ---
images = []
labels = []
label_map = {}

for idx, person_name in enumerate(sorted(os.listdir(dataset_dir))):
    person_dir = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_dir):
        label_map[idx] = person_name
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.pgm', '.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (100, 100))
                images.append(img_resized)
                labels.append(idx)

images = np.array(images)
labels = np.array(labels)

# --- LBP Feature Extraction ---
def compute_lbp_hist(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    return hist

lbp_features = np.array([compute_lbp_hist(img) for img in images])

# --- HOG Feature Extraction ---
def compute_hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

hog_features = np.array([compute_hog_features(img) for img in images])

# --- Combine LBP + HOG Features ---
X_combined = np.hstack((lbp_features, hog_features))

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# --- PCA Dimensionality Reduction ---
pca = PCA(n_components=0.98, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Neural Network Classifier ---
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    verbose=True
)


mlp.fit(X_train, y_train)

# --- Prediction & Accuracy ---
y_pred = mlp.predict(X_test)

# --- Classification Report ---
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=[label_map[i] for i in np.unique(labels)], zero_division=0))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_map[i] for i in np.unique(labels)],
            yticklabels=[label_map[i] for i in np.unique(labels)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Neural Network Model")
plt.tight_layout()
plt.show()

# --- Save Final Model and Preprocessors ---
joblib.dump(mlp, r"C:\Face Recognition Dataset\olivetti dataset\nn\nn_final_model.pkl")
joblib.dump(scaler, r"C:\Face Recognition Dataset\olivetti dataset\nn\scaler.pkl")
joblib.dump(pca, r"C:\Face Recognition Dataset\olivetti dataset\nn\pca.pkl")
joblib.dump(label_map, r"C:\Face Recognition Dataset\olivetti dataset\nn\label_map.pkl")

print("Neural Network model, scaler, PCA, and label map saved successfully.")
