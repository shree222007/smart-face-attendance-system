# This is the SVM model code for the ORL Dataset
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Path to the extracted AT&T Olivetti dataset
dataset_dir = r"C:\Face Recognition Dataset\olivetti dataset\att_faces"

# Load images and labels
images = []
labels = []

for label in range(40):  # s1 to s40
    person_dir = os.path.join(dataset_dir, f's{label+1}')
    for filename in os.listdir(person_dir):
        if filename.endswith(".pgm"):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(label)

images = np.array(images)
labels = np.array(labels)

n_samples, h, w = images.shape

# Flatten images for PCA
X_flat = images.reshape(n_samples, -1)

# Apply PCA for dimensionality reduction
n_components = 80
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_flat)

# Function to compute LBP histogram
def compute_lbp_hist(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    return hist

# Compute LBP features for all images
lbp_features = np.array([compute_lbp_hist(img) for img in images])

# Combine PCA and LBP features
X_combined = np.hstack([X_pca, lbp_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42)

# SVM classifier
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
