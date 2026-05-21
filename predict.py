import os
import json
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================================
# CURRENT DIRECTORY
# =====================================================
print("CURRENT FOLDER:", os.getcwd())
print("\nFILES IN FOLDER:", os.listdir())

# =====================================================
# LOAD CONFIG  ← written by train.py
# =====================================================
config_path = "model_config.json"

if not os.path.exists(config_path):
    print("\n❌ model_config.json not found. Run train.py first.")
    exit(1)

with open(config_path) as f:
    cfg = json.load(f)

print(f"\n✅ Config loaded")
print(f"   Architecture : {cfg['architecture']}")
print(f"   Input size   : {cfg['input_size']}×{cfg['input_size']}")
print(f"   Classes      : {cfg['num_classes']}")
print(f"   Val accuracy : {cfg['val_accuracy']}%")

# =====================================================
# IMAGE PATH  ← change this to your test image
# =====================================================
img_path = r"C:\Users\KIIT\OneDrive\Desktop\Leafdiseasedetection\1ec3e679-2b1e-4243-9e62-67f8f04314b3___FREC_C.Rust 4328.JPG"

# =====================================================
# PATHS FROM CONFIG
# =====================================================
model_path  = cfg["model_path"]
labels_path = cfg["labels_path"]
input_size  = cfg["input_size"]

# =====================================================
# CHECK FILES
# =====================================================
print("\nChecking image  :", os.path.exists(img_path))
print("Checking model  :", os.path.exists(model_path))
print("Checking labels :", os.path.exists(labels_path))

# =====================================================
# LOAD MODEL
# =====================================================
print("\nLoading model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded")

# =====================================================
# LOAD LABELS
# =====================================================
with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print("✅ Labels loaded")

# =====================================================
# LOAD + PREPROCESS IMAGE
# =====================================================
img = image.load_img(img_path, target_size=(input_size, input_size))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)          # MobileNetV2 preprocessor
img_array = np.expand_dims(img_array, axis=0)

# =====================================================
# PREDICTION
# =====================================================
print("\nRunning prediction...")
prediction    = model.predict(img_array)
predicted_idx = np.argmax(prediction)
pred_class    = class_names[predicted_idx]
confidence    = float(np.max(prediction) * 100)

# Top 3
top3 = prediction[0].argsort()[-3:][::-1]

# =====================================================
# RESULT
# =====================================================
print("\n========== RESULT ==========\n")
print(f"Prediction : {pred_class}")
print(f"Confidence : {confidence:.2f}%")
print("\nTop 3 predictions:")
for i in top3:
    print(f"  {class_names[i]:<45} {prediction[0][i]*100:.2f}%")
print("\n============================")