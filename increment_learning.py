import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ======================
# LOAD MODEL AND LABELS
# ======================
OUTPUT_PATH = "output"
MODEL_PATH = os.path.join(OUTPUT_PATH, "emotion_recognition_model.keras")
LABELS_PATH = os.path.join(OUTPUT_PATH, "labels.npy")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels not found at {LABELS_PATH}")

print("[INFO] Loading trained model and labels...")
model = load_model(MODEL_PATH)
classes = np.load(LABELS_PATH, allow_pickle=True)
encoder = LabelEncoder()
encoder.fit(classes)
print(f"[INFO] Loaded emotion classes: {encoder.classes_}")

# ======================
# FUNCTION TO EXTRACT MFCC
# ======================
def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc_scaled, axis=(0, 2))  # shape=(1, 40, 1)

# ======================
# FUNCTION TO UPDATE MODEL OUTPUT
# ======================
def expand_model_output(model, new_num_classes):
    penultimate_layer = model.layers[-2].output
    new_output = Dense(new_num_classes, activation='softmax', name='updated_output')(penultimate_layer)
    new_model = Model(inputs=model.input, outputs=new_output)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model

# ======================
# REAL-TIME PREDICTION LOOP
# ======================
while True:
    audio_path = input("\n[INPUT] Enter audio file path (or 'quit' to exit): ").strip()
    if audio_path.lower() == 'quit':
        break
    if not os.path.exists(audio_path):
        print("[WARNING] File not found. Try again.")
        continue

    X_input = extract_features(audio_path)
    pred_probs = model.predict(X_input)
    pred_class_index = np.argmax(pred_probs)
    pred_emotion = encoder.inverse_transform([pred_class_index])[0]
    print(f"[INFO] Predicted emotion: {pred_emotion}")

    correct = input("[INPUT] Is the prediction correct? (y/n): ").strip().lower()
    if correct == 'y':
        continue
    else:
        correct_emotion = input("[INPUT] Enter the correct emotion: ").strip()

        if correct_emotion not in encoder.classes_:
            print("[INFO] Adding new emotion class:", correct_emotion)
            # Update encoder
            new_classes = np.append(encoder.classes_, correct_emotion)
            encoder.fit(new_classes)
            np.save(LABELS_PATH, encoder.classes_)
            # Update model output layer
            model = expand_model_output(model, new_num_classes=len(encoder.classes_))
            print(f"[INFO] Model output layer expanded for {len(encoder.classes_)} classes.")

        # Prepare one-hot target
        y_correct = encoder.transform([correct_emotion])
        y_correct_onehot = to_categorical(y_correct, num_classes=len(encoder.classes_))

        # Incremental learning
        print("[INFO] Fine-tuning model with corrected label...")
        model.fit(X_input, y_correct_onehot, epochs=3, verbose=1)

        # Save updated model
        model.save(MODEL_PATH)
        print("[INFO] Model and labels updated and saved.")
