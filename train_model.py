import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================
# CONFIGURATION
# ======================
print("[INFO] Starting Emotion Recognition Training Module...")

# Enable GPU (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("[INFO] GPU found and enabled:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[INFO] No GPU found. Running on CPU.")

# Dataset directory
DATASET_PATH = "dataset"

# Ensure output directory exists
OUTPUT_PATH = "output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("[INFO] Scanning dataset directory...")

# ======================
# DATASET LOADING
# ======================
labels = []
features = []
emotion_folders = sorted(os.listdir(DATASET_PATH))
print(f"[INFO] Found emotion folders: {emotion_folders}")

for emotion in emotion_folders:
    folder_path = os.path.join(DATASET_PATH, emotion)
    if not os.path.isdir(folder_path):
        continue
    print(f"[INFO] Processing emotion: {emotion}")
    for file in os.listdir(folder_path):
        if not file.endswith(('.wav', '.mp3')):
            continue
        file_path = os.path.join(folder_path, file)
        try:
            audio, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            features.append(mfcc_scaled)
            labels.append(emotion)
        except Exception as e:
            print(f"[WARNING] Could not process {file_path}: {e}")

print(f"[INFO] Total samples collected: {len(features)}")

# ======================
# DATA PREPARATION
# ======================
print("[INFO] Encoding labels and preparing training data...")

X = np.array(features)
y = np.array(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Reshape for Conv1D input
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print(f"[INFO] Training samples: {X_train.shape[0]}")
print(f"[INFO] Testing samples: {X_test.shape[0]}")
print(f"[INFO] Input shape: {X_train.shape[1:]}")

# ======================
# MODEL BUILDING
# ======================
print("[INFO] Building Conv1D model...")

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ======================
# MODEL TRAINING
# ======================
print("[INFO] Starting training...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ======================
# PERFORMANCE PLOTS
# ======================
print("[INFO] Saving accuracy and loss graphs...")

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, "accuracy_plot.png"))

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, "loss_plot.png"))

# ======================
# MODEL SAVING
# ======================
print("[INFO] Saving trained model...")

model.save(os.path.join(OUTPUT_PATH, "emotion_recognition_model.keras"))

np.save(os.path.join(OUTPUT_PATH, "labels.npy"), encoder.classes_)
print(f"[INFO] Label classes saved as '{os.path.join(OUTPUT_PATH, 'labels.npy')}'")

print(f"[INFO] Model and plots saved in '{OUTPUT_PATH}' folder")
print("[INFO] Training complete.")