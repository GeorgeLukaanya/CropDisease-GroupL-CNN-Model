## 🌿 Plant Disease Classification Model — README

This project implements a deep learning pipeline for detecting and classifying plant leaf diseases using a Convolutional Neural Network (CNN). The system is designed to assist farmers with early disease detection and treatment recommendations. It forms part of a larger intelligent platform aiming to digitize plant health diagnostics with AI.

---

### 🧠 Model Overview (Current Implementation)

#### Model Architecture

The core classifier is a **deep CNN** with the following structure:

* **Convolutional Stack**:

  * Conv2D Layers: 32 → 64 → 128 → 128 filters (ReLU activations)
  * BatchNormalization after each Conv2D
  * MaxPooling2D for spatial downsampling

* **Dense Layers**:

  * Flatten → Dense(512) → Dropout(0.5)
  * Dense(256) → Dropout(0.3)
  * Output Layer: Dense(38, softmax)

* **Optimizer**: `Adam`, Learning rate = 0.001

* **Loss**: `Categorical Crossentropy` (multiclass classification)

---

### 🗂 Dataset and Preprocessing

* **Dataset**: *New Plant Diseases Dataset (Augmented)*

  * \~70,000 images across 38 disease classes
  * Augmented with rotation, zoom, flipping, shear, and shift
* **Image Format**: RGB, resized to 224×224 pixels
* **Image Validation**: Corrupted image detection and logging via PIL
* **Data Generators**:

  * `ImageDataGenerator` for train/valid/test with real-time augmentation
  * Normalization to \[0,1] pixel scale

---

### ⚙️ Training Process

* **Batch Size**: 32 (auto-adjusts on failure)
* **Epochs**: Up to 15
* **Early Stopping**: Monitors `val_loss`, patience = 5
* **Class Weights**: Automatically calculated for class imbalance
* **Callbacks**:

  * `ModelCheckpoint` (best\_model.keras)
  * EarlyStopping (best weight restoration)

---

### 📈 Evaluation and Visualizations

* **Metrics**:

  * Classification Report (Precision, Recall, F1)
  * Confusion Matrix
  * Accuracy/Loss Curves
* **Tools**:

  * `Matplotlib`, `Seaborn`, and `Plotly` for plots
  * Outputs: `confusion_matrix.png`, `training_history.png`, `class_distribution.png`
* **Test Fallback**: If test set fails, validation set is used for evaluation

---

### 📝 System Artifacts

* `trained_cnn_model.keras` — Final trained model
* `best_model.keras` — Best model during training
* `class_indices.txt` — Class label mappings (for integration with Django API)
* `training.log` — Complete training process logs

---

## 🚧 Current Limitations

Despite achieving \~86% accuracy in lab conditions, the system faces challenges in real-world use:

* **Background Noise**: Soil, fingers, tools, or other distractions confuse the model
* **Multi-Leaf Images**: Cannot handle multiple leaves or plant sections in one image
* **Flat Output**: No severity or progression scoring
* **Assumes Single, Clear Leaf** per image

---

## 🛠️ Proposed Solution — Two-Stage Pipeline

To improve robustness in practical scenarios, we propose a **multi-model architecture**:

### Stage 1: Object Detection with YOLOv8

* Detects and crops **individual leaves** from complex images
* Removes clutter such as sky, soil, or tools
* Reduces false predictions due to irrelevant content

### Stage 2: CNN Classification

* Applies the current classifier only to **clean leaf crops**
* Maintains softmax-based confidence scoring
* Focuses predictions solely on the leaf

---

### 🌡️ Future: Severity Scoring + Co-Infection Detection

A **Vision Transformer (ViT)** model can be added to classify disease severity (mild, moderate, severe), enabling:

* **Triage & Escalation**: Flagging high-risk infections
* **Multiple Label Output**: For co-infections and overlapping conditions

---

### 🔁 Technical Comparison

| Feature            | Current System       | Proposed Pipeline           |
| ------------------ | -------------------- | --------------------------- |
| Models Used        | 1 CNN                | YOLOv8 + CNN + ViT          |
| Input Format       | Leaf-only            | Raw plant image             |
| Output             | Disease + confidence | BBox + Disease + Severity   |
| Inference Latency  | 120–250 ms (CPU)     | 500–800 ms (GPU optimized)  |
| Estimated Accuracy | \~86%                | Up to 93–95% with detection |

---

### ⚙️ Operational Considerations

* **Runtime Flow**:
  `yolo.detect()` → `crop_leaf()` → `cnn.predict()` → `score_severity()`
* **UI Improvements**:

  * Bounding box overlays
  * Severity bars
  * Confidence-level indicators
  * Multi-disease labels

---

### ✅ Best Practices

| Category      | Recommendation                            |
| ------------- | ----------------------------------------- |
| MLOps         | Config/model separation, CI/CD ready      |
| Security      | PIL-based image checks, silent error logs |
| Accessibility | Confidence labels, fallback suggestions   |

---

