import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# DATASET PATH
# =====================================================
dataset_path = "PlantVillage/train"

# =====================================================
# IMAGE SETTINGS
# =====================================================
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15

# =====================================================
# DATA PREPROCESSING
# =====================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# =====================================================
# TRAINING DATA
# =====================================================
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# =====================================================
# VALIDATION DATA
# =====================================================
val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =====================================================
# SAVE LABELS
# =====================================================
print("\nClass Indices:")
print(train_data.class_indices)

class_names = list(train_data.class_indices.keys())

with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n✅ labels.txt saved successfully")

# =====================================================
# LOAD PRETRAINED MODEL
# =====================================================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# =====================================================
# FREEZE BASE MODEL
# =====================================================
base_model.trainable = False

# =====================================================
# BUILD MODEL
# =====================================================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# =====================================================
# COMPILE
# =====================================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# EARLY STOPPING
# =====================================================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

model.summary()

# =====================================================
# TRAIN
# =====================================================
print("\n==============================")
print("STARTING TRAINING")
print("==============================\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =====================================================
# FINAL ACCURACY
# =====================================================
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc   = history.history['val_accuracy'][-1] * 100

print(f"\nTraining Accuracy   : {final_train_acc:.2f}%")
print(f"Validation Accuracy : {final_val_acc:.2f}%")

# =====================================================
# SAVE MODEL
# =====================================================
model_save_path = "leaf_disease_model.keras"
model.save(model_save_path)
print(f"\n✅ Model saved → {model_save_path}")

# =====================================================
# SAVE CONFIG  ← NEW: app.py + predict.py read this
# =====================================================
config = {
    "model_path":       model_save_path,
    "labels_path":      "labels.txt",
    "architecture":     "MobileNetV2",
    "preprocessor":     "mobilenet_v2",
    "input_size":       IMG_SIZE,
    "num_classes":      train_data.num_classes,
    "train_accuracy":   round(final_train_acc, 2),
    "val_accuracy":     round(final_val_acc,   2),
    "epochs_trained":   len(history.history['accuracy']),
    "batch_size":       BATCH_SIZE,
    "dataset_path":     dataset_path,
    "class_names":      class_names
}

with open("model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ model_config.json saved — app.py and predict.py will use this automatically")

# =====================================================
# PLOT
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("accuracy_graph.png")
plt.show()
print("\n✅ Graph saved → accuracy_graph.png")