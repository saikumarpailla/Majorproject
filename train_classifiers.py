import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pickle
import tensorflow.keras.backend as K

# Custom metrics for CNN
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# HOG feature extraction
def hog_feature_extraction(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = cv2.equalizeHist(image)
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Load CNN model and feature extractor
cnn_model = load_model('xec.h5', custom_objects={'f1_score': f1_m, 'precision_score': precision_m, 'recall_score': recall_m}, compile=False)
cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Load dataset
dataset_path = r'C:\Users\Paill\Downloads\code floder\code floder\Extension\signatures'
X_hog, X_hybrid, y = [], [], []
for label, folder in enumerate(['forgery', 'genuine']):  # 0 = Forgery, 1 = Genuine
    folder_path = os.path.join(dataset_path, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        # HOG features
        hog_features = hog_feature_extraction(img_path)
        X_hog.append(hog_features)
        
        # CNN features
        img = load_img(img_path, target_size=(128, 128))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        cnn_features = cnn_feature_extractor.predict(img).flatten()
        
        # Hybrid features
        combined_features = np.concatenate([cnn_features, hog_features])
        X_hybrid.append(combined_features)
        
        y.append(label)

X_hog = np.array(X_hog)
X_hybrid = np.array(X_hybrid)
y = np.array(y)

# Train HOG classifier
hog_classifier = SVC(kernel='linear', probability=True)
hog_classifier.fit(X_hog, y)
with open('hog_classifier.pkl', 'wb') as f:
    pickle.dump(hog_classifier, f)
print("HOG classifier trained and saved!")

# Train Hybrid classifier
hybrid_classifier = SVC(kernel='linear', probability=True)
hybrid_classifier.fit(X_hybrid, y)
with open('hybrid_classifier.pkl', 'wb') as f:
    pickle.dump(hybrid_classifier, f)
print("Hybrid classifier trained and saved!")
