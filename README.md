# Gesture Recognition Model

This project implements a **gesture recognition system** using a deep learning pipeline. The model leverages sequential data from **body landmarks** extracted with [Mediapipe](https://developers.google.com/mediapipe) and employs **Bidirectional GRU (Gated Recurrent Units)** for robust temporal pattern recognition.

---

## 📌 Key Features
- **Input Data:**  
  - 45 frames per gesture  
  - 258 features per frame (landmarks from **pose** and **hands only**)  
- **Model Architecture:**  
  - Two Dense (DNN) layers for feature extraction  
  - Three Bidirectional GRU layers for temporal modeling  
  - Batch Normalization layers for improved training stability  
  - Final Dense layers for classification  
- **Output:**  
  - 6–8 gesture classes (depending on dataset configuration)  
- **Visualization:**  
  - Real-time prediction with probability visualization for gesture confidence  

---

## 📂 Dataset Preparation
**Data Source:**
- Extracted Mediapipe landmarks:  
  - Pose: 33 points  
  - Hands: 21 points each  
- Face landmarks excluded to improve clarity and reduce confusion  

**Preprocessing:**
- Gesture sequence → NumPy array of shape `(45, 258)`  
- Normalization for consistent input  
- Optional **PCA** for dimensionality reduction  

---

## 🧠 Model Overview
The model balances complexity and performance, achieving optimal accuracy even on limited hardware.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, BatchNormalization, Bidirectional

model = Sequential()

# DNN Layers
model.add(Dense(256, activation='relu', input_shape=(45, 258)))
model.add(Dense(128, activation='relu'))

# Bidirectional GRU Layers
model.add(Bidirectional(GRU(256, return_sequences=True)))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(Bidirectional(GRU(64)))

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='softmax'))  # For 8 gesture classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## ⚙️ Training Details

* **Batch Size:** 4 (optimized for memory & convergence)
* **Learning Rate:** `1e-4` with Adam optimizer
* **Epochs:** 100 for sufficient learning without overfitting

---

## 🎥 Real-Time Prediction Pipeline

1. Capture real-time video via **OpenCV**
2. Extract keypoints per frame using **Mediapipe**
3. Predict gestures using the latest 45 frames

**Visualization:**

* Probability bars for all gesture classes
* Top prediction displayed on video feed

---

## 🚧 Challenges & Solutions

* **Face Landmark Overload:**

  * Initially included \~1400 static features → caused confusion
  * **Solution:** Removed face landmarks; focused on pose & hands
* **Hardware Limitations:**

  * Optimized model size and preprocessing for reasonable speed
* **Data Quality:**

  * Higher-quality data → significant accuracy improvement

---

## 📊 Performance

* **Training Accuracy:** High accuracy achieved with refined dataset
* **Inference Speed:** \~19ms/step (real-time capable)

---

## 🔮 Future Work

* Add **data augmentation** for robustness
* Apply **quantization & pruning** for faster edge inference
* Expand gesture classes for more use cases

---

## 🛠 Requirements

* Python 3.7+
* TensorFlow 2.x
* Mediapipe
* OpenCV
* NumPy

