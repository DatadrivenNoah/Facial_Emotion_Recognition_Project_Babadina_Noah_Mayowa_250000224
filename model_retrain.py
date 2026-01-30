"""
model_retrain.py
Train a CNN model on FER2013 emotion detection dataset
Safe for modern TensorFlow / Streamlit deployment
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

# =========================
# GLOBAL SETTINGS
# =========================
EMOTIONS = [
    'Angry', 'Disgusted', 'Fearful',
    'Happy', 'Neutral', 'Sad', 'Surprised'
]

EMOTION_MAP = {i: emotion for i, emotion in enumerate(EMOTIONS)}
MODEL_NAME = "emotion_model.keras"

# Clear previous TF sessions (important)
tf.keras.backend.clear_session()


# =========================
# DATA HANDLING
# =========================
def download_fer2013(data_dir="data"):
    print("FER2013 must be downloaded manually from Kaggle.")
    print("https://www.kaggle.com/datasets/msambare/fer2013")

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    return data_path / "fer2013.csv"


def load_fer2013_data(csv_path, test_size=0.2, val_size=0.2):
    if not csv_path.exists():
        raise FileNotFoundError(
            "fer2013.csv not found. Place it inside the 'data/' folder."
        )

    print("Loading FER2013 dataset...")
    df = pd.read_csv(csv_path)

    X, y = [], []

    for i, row in df.iterrows():
        pixels = np.array(row["pixels"].split(), dtype="uint8").reshape(48, 48)
        X.append(pixels)
        y.append(row["emotion"])

        if i % 5000 == 0:
            print(f"Processed {i}/{len(df)} images")

    X = np.array(X, dtype="float32") / 255.0
    X = np.expand_dims(X, axis=-1)
    y = keras.utils.to_categorical(y, num_classes=7)

    from sklearn.model_selection import train_test_split

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=test_size / (test_size + val_size),
        random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =========================
# MODEL ARCHITECTURE
# =========================
def build_emotion_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(7, activation="softmax")
    ])

    return model


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =========================
# TRAINING
# =========================
def train_model(model, train_data, val_data, epochs=50, batch_size=32):
    X_train, y_train = train_data
    X_val, y_val = val_data

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            "emotion_model_best.keras",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history


# =========================
# EVALUATION & SAVING
# =========================
def evaluate_model(model, test_data):
    X_test, y_test = test_data
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test Accuracy: {acc*100:.2f}%")
    return loss, acc


def save_model(model):
    model.save(MODEL_NAME)
    print(f"✓ Model saved as {MODEL_NAME}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("FER2013 Emotion Detection Training")
    print("=" * 60)

    csv_path = download_fer2013()

    train_data, val_data, test_data = load_fer2013_data(csv_path)

    model = build_emotion_model()
    model = compile_model(model)

    model.summary()

    history = train_model(model, train_data, val_data, epochs=50)

    evaluate_model(model, test_data)

    save_model(model)

    print("\n✓ Training complete")
    print("✓ Model ready for Streamlit / Render")
