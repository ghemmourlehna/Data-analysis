# Real vs Fake Face Detection using Manual PCA and Machine Learning

## Overview

This project focuses on **face anti-spoofing detection**, also known as **Face Liveness Detection**.
The goal is to classify facial images into two categories:

* **Real / Live**: a genuine face of a physically present person
* **Fake / Spoof**: an attack using a printed photo, replayed video, screen display, or partial 3D mask

The project was developed as part of a **Data Analysis module** and emphasizes the manual implementation of **Principal Component Analysis (PCA)** using **NumPy**, without relying on automated PCA functions from machine learning libraries.

After reducing the dimensionality of face images using PCA, several classification approaches were evaluated:

* PCA + KNN
* PCA + SVM with RBF kernel
* Hybrid PCA + CNN model

The best performance was achieved using the hybrid model, combining global information extracted by PCA with local texture features learned by a CNN.

---

## Project Objectives

The main objectives of this project are:

* Understand the problem of facial spoofing attacks
* Work with a real-world anti-spoofing dataset
* Implement PCA manually from its mathematical foundations
* Reduce the dimensionality of facial images
* Visualize eigenfaces and explained variance
* Train classical machine learning classifiers on PCA features
* Compare KNN and SVM performances
* Build a hybrid model combining PCA and CNN features
* Evaluate models using accuracy, confusion matrix, and ROC-AUC

---

## Problem Statement

Modern biometric systems often rely on facial recognition for authentication.
However, these systems can be vulnerable to spoofing attacks, where an attacker tries to deceive the system using:

* Printed face photos
* Replayed videos on a screen
* Photos with eye cutouts
* Partial 3D masks

Face anti-spoofing aims to determine whether a face presented to the camera belongs to a real person or to an artificial presentation attack.

This project addresses the following question:

> Can we distinguish real faces from spoofed faces using manual PCA-based feature extraction and machine learning classifiers?

---

## Dataset

The dataset used in this project is **CelebA-Spoof**, a large-scale public dataset designed for facial anti-spoofing research.

### Dataset Characteristics

* More than 600,000 images
* 10,177 unique subjects
* Real and spoofed face images
* Multiple attack types
* Different lighting conditions and environments

For this project, a subset of the dataset was used.

### Data Used

| Usage            |         Size | Image Format           |
| ---------------- | -----------: | ---------------------- |
| PCA + KNN + SVM  | 1,500 images | 64×64 grayscale images |
| Hybrid CNN model | 3,000 images | 128×128 RGB images     |
| Final test set   |   300 images | Depending on the model |

The PCA-based approach used **1,500 images**, with the following class distribution:

* **524 real images**
* **976 fake images**

This imbalance was handled during classification, especially in the SVM model using class balancing.

---

## Image Preprocessing

Before applying PCA, each image was preprocessed using the following steps:

1. Read the image in grayscale
2. Resize it to **64×64 pixels**
3. Flatten the image into a vector of **4,096 features**
4. Normalize pixel values to the range `[0, 1]`

Each image is therefore represented as a vector:

```text
64 × 64 = 4096 features
```

---

## Manual PCA Implementation

One of the main contributions of this project is the complete manual implementation of PCA using NumPy.

The PCA pipeline was implemented in 8 steps:

### 1. Mean Face Computation

The average face is computed from all images and used as a reference for centering the data.

```python
X_mean = np.mean(X_acp, axis=0)
```

### 2. Data Centering

The mean face is subtracted from each image.

```python
X_centered = X_acp - X_mean
```

### 3. Standardization

Each pixel feature is divided by its standard deviation.

```python
std_acp = np.std(X_centered, axis=0) + 1e-8
X_reduced = X_centered / std_acp
```

### 4. Correlation Matrix Computation

The correlation matrix is computed manually.

```python
R = (X_reduced.T @ X_reduced) / n
```

### 5. Eigenvalue Decomposition

Eigenvalues and eigenvectors are computed using NumPy.

```python
eigenvalues, eigenvectors = np.linalg.eigh(R)
```

### 6. Sorting Principal Components

The eigenvectors are sorted according to descending eigenvalues.

```python
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

### 7. Explained Variance and Choice of k

The explained variance was used to choose the number of principal components.

The selected value was:

```text
k = 200 principal components
```

This preserves approximately:

```text
91.4% of the total variance
```

### 8. Projection into PCA Space

The original data is projected into the reduced PCA space.

```python
k = 200
V_k = eigenvectors[:, :k]
T = X_reduced @ V_k
```

The final representation of each image becomes:

```text
4096 dimensions → 200 dimensions
```

---

## PCA Interpretation

The PCA analysis provides several important insights:

* The first components capture global variations such as lighting and pose.
* Later components capture finer facial details.
* Eigenfaces allow visual interpretation of the main variation patterns in the dataset.
* PCA reduces noise while preserving the most important information.
* PCA improves the efficiency of classical classifiers such as KNN and SVM.

---

## Classification Models

### 1. KNN Classifier

A K-Nearest Neighbors classifier was trained using the PCA-reduced features.

Configuration:

```python
KNeighborsClassifier(n_neighbors=5, metric="euclidean")
```

Result:

```text
Accuracy: 84.33%
```

KNN provided a strong baseline, showing that the PCA space is sufficiently discriminative for real/fake classification.

---

### 2. SVM with RBF Kernel

A Support Vector Machine classifier with RBF kernel was also trained on the PCA features.

Configuration:

```python
SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)
```

Results:

```text
Accuracy: 85.67%
AUC ROC: 0.9081
```

The SVM outperformed KNN thanks to its ability to model non-linear decision boundaries.

---

### 3. Hybrid PCA + CNN Model

The hybrid model combines two complementary branches:

### PCA Branch

This branch uses the 200-dimensional PCA representation.

```text
PCA features → Dense(128) → Dense(64)
```

### CNN Branch

This branch processes RGB images of size 128×128.

```text
128×128 RGB image
→ Conv2D(32)
→ MaxPooling
→ Conv2D(64)
→ MaxPooling
→ Conv2D(128)
→ Global Average Pooling
→ Dense(256)
```

### Fusion

The two branches are concatenated.

```text
PCA branch + CNN branch
→ Dense(128)
→ Dropout(0.5)
→ Dense(64)
→ Dense(1, sigmoid)
```

This hybrid approach combines:

* Global structural information from PCA
* Local texture and artifact features from CNN

---

## Training Configuration

The hybrid model was trained using the following parameters:

| Parameter      | Value               |
| -------------- | ------------------- |
| Optimizer      | Adam                |
| Learning rate  | 0.001               |
| Loss function  | Binary Crossentropy |
| Batch size     | 32                  |
| Maximum epochs | 50                  |
| Early stopping | Yes                 |
| GPU            | NVIDIA Tesla T4     |

---

## Results

### Performance Comparison

| Model            | Accuracy | AUC ROC | Notes                |
| ---------------- | -------: | ------: | -------------------- |
| PCA + KNN        |   84.33% |       — | Baseline classifier  |
| PCA + SVM RBF    |   85.67% |  0.9081 | Best classical model |
| Hybrid PCA + CNN |   94.67% |  0.9919 | Best overall model   |

The hybrid PCA + CNN model achieved the best results, with a significant improvement over the classical PCA-based models.

---

## Confusion Matrix – Hybrid Model

On the final test set of 300 images, the hybrid model obtained:

| Prediction Result | Count |
| ----------------- | ----: |
| True Negatives    |   189 |
| True Positives    |    90 |
| False Positives   |     6 |
| False Negatives   |    15 |

This gives an accuracy of approximately:

```text
93.0% on the final test sample
```

The model is especially effective at detecting spoofed faces, which is important in security-sensitive biometric systems.

---

## ROC Curve

The hybrid model achieved a strong ROC-AUC score:

```text
AUC = 0.9741 on the final ROC curve
```

This means the model has a very strong ability to separate real and fake faces.

The comparison between SVM and the hybrid model shows that the hybrid model provides better performance across almost all false positive rate values.

---

## Key Findings

The main findings of this project are:

* PCA is effective for dimensionality reduction and visual interpretation.
* A reduction from 4,096 to 200 dimensions preserves most of the useful information.
* Classical classifiers such as KNN and SVM achieve good results using PCA features.
* SVM performs better than KNN due to its non-linear decision boundary.
* CNNs capture local texture details that PCA alone cannot extract.
* Combining PCA and CNN features gives the best overall performance.
* The hybrid model significantly improves accuracy and AUC.

---

## Project Structure

```text
real-vs-fake-face-detection/
│
├── data/
│   └── CelebA-Spoof subset
│
├── notebooks/
│   └── data_analysis_project.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── manual_pca.py
│   ├── classification.py
│   └── hybrid_model.py
│
├── results/
│   ├── eigenfaces.png
│   ├── scree_plot.png
│   ├── pca_projection.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── report/
│   └── Rapport_AD.pdf
│
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/real-vs-fake-face-detection.git
cd real-vs-fake-face-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

On Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

The project uses the following main libraries:

```text
numpy
opencv-python
matplotlib
scikit-learn
tensorflow
keras
pandas
seaborn
```

---

## How to Run

### Run PCA and classical classifiers

```bash
python src/manual_pca.py
python src/classification.py
```

### Train the hybrid PCA + CNN model

```bash
python src/hybrid_model.py
```

### Run the notebook

You can also run the full project from the Jupyter notebook:

```bash
jupyter notebook notebooks/data_analysis_project.ipynb
```

---

## Visualizations

The project includes several visualizations:

* Mean face
* Centered and standardized images
* Eigenfaces
* Scree plot
* Cumulative explained variance
* PCA reconstruction
* Real/Fake separation in PCA space
* 3D PCA projection
* Correlation circle
* Confusion matrix
* ROC curves

---

## Future Improvements

Possible improvements include:

* Using a larger subset of the CelebA-Spoof dataset
* Applying data augmentation to improve generalization
* Testing additional classifiers such as Random Forest or XGBoost
* Fine-tuning a pre-trained CNN model
* Improving class imbalance handling
* Deploying the model as a web application
* Adding real-time webcam-based face anti-spoofing detection

---

## Academic Context

This project was developed for the **Data Analysis** module in **Master 1 Visual Computing**.

---



## License

This project is intended for academic and research purposes.
