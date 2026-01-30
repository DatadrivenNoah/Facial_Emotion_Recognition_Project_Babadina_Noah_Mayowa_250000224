"""
model_saving.py
Safe Emotion Detection Model Builder
(Keras 3, Streamlit & Render compatible)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# --------------------------------------------------
# SAFE CUSTOM LAYERS (NO LAMBDA)
# --------------------------------------------------
class GrayscaleToRGB(layers.Layer):
    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)

    def get_config(self):
        return super().get_config()


class MobileNetPreprocess(layers.Layer):
    def call(self, inputs):
        return keras.applications.mobilenet_v2.preprocess_input(inputs)

    def get_config(self):
        return super().get_config()


# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------
def build_emotion_model():
    print("Building MobileNetV2 emotion model (SAFE)...")

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        layers.Resizing(224, 224),
        GrayscaleToRGB(),
        MobileNetPreprocess(),

        base_model,
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(7, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# --------------------------------------------------
# SAVE & VERIFY
# --------------------------------------------------
def save_and_verify(model, name="emotion_model.keras"):
    model.save(name)
    print(f"\n✓ Model saved as {name}")

    tf.keras.models.load_model(
        name,
        custom_objects={
            "GrayscaleToRGB": GrayscaleToRGB,
            "MobileNetPreprocess": MobileNetPreprocess
        }
    )
    print("✓ Model reload successful")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Emotion Detection Model Builder (SAFE VERSION)")
    print("=" * 60)

    model = build_emotion_model()
    save_and_verify(model)
