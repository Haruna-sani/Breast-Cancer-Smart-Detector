
# 🧠 Breast Cancer Detection Using CNN on Ultrasound Images

This repository presents a **Convolutional Neural Network (CNN)** model developed to classify **breast ultrasound images** into two categories: **Benign** and **Malignant**. The goal is to aid in the early detection of breast cancer using deep learning techniques.

---

## 📁 Dataset

- **Source**: [BUS - Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/orvile/bus-uc-breast-ultrasound)
- The dataset contains 3 classes: **Benign**, **Malignant**, and **Normal**, but this project focuses on a binary classification (**Benign** vs **Malignant**).
- Each image was resized to **128 × 128** pixels with **3 RGB channels**.

---

## ⚙️ Input Shape & Configuration

```python
INPUT_SHAPE = (128, 128, 3)
KERNEL_SIZE = (3, 3)
```

---

## 🧠 Model Architecture

The CNN model consists of multiple convolutional layers with ReLU activation, followed by batch normalization, max pooling, and dropout for regularization. A fully connected dense layer with a softmax activation is used for final classification.

```python
model = Sequential()

# First Convolution Block
model.add(Conv2D(128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second Convolution Block
model.add(Conv2D(64, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third Convolution Block
model.add(Conv2D(128, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))  # Output layer
```

---

## 📊 Data Preprocessing & Augmentation

To improve model generalization and prevent overfitting, the training set was augmented using the following parameters:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 64
data_generator = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = data_generator.flow(X_train, y_train, batch_size=batch_size)
```

- One-hot encoding of labels was performed to classify two categories:
```python
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=2)
```

- Train-test split: **80% for training**, **20% for testing**

---

## 🧪 Model Evaluation

The model achieved the following results on the test dataset:

### 🧾 Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Benign      | 0.62      | 0.78   | 0.69     | 74      |
| Malignant   | 0.77      | 0.60   | 0.67     | 89      |
| **Accuracy**|           |        | **0.68** | **163** |
| Macro Avg   | 0.69      | 0.69   | 0.68     | 163     |
| Weighted Avg| 0.70      | 0.68   | 0.68     | 163     |

✅ **Overall Accuracy: 68%**

---

## 🔧 Tools & Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

---

## 📌 Future Improvements

- Use pre-trained models (Transfer Learning) like VGG16, ResNet, or EfficientNet
- Experiment with deeper CNN architectures
- Apply class balancing or oversampling
- Hyperparameter tuning with cross-validation

---

## 📢 Disclaimer

This project is for **educational and research purposes** only and should not be used as a substitute for professional medical diagnosis.

---

## 🙌 Acknowledgements

Thanks to Kaggle and the authors of the **BUS dataset** for providing valuable open-access data for breast cancer research.
