# Amazon Fashion Products Classification

**This project implements an image classification model for Amazon fashion products using TensorFlow and MobileNetV2. The model predicts the category of a product based on its image.**

---

## **Table of Contents**
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

---

## **Dataset**

The dataset contains **13,156 entries** with product images and metadata:

| Column       | Description                                      |
|--------------|--------------------------------------------------|
| product_id   | Unique identifier for each product              |
| brand        | Brand name of the product                        |
| title        | Product title                                   |
| price        | Product price                                   |
| category     | Product category (target for classification)   |
| rating       | Product rating                                  |
| image_url    | URL of product image                             |
| product_url  | URL of product page on Amazon                    |

**Sample Data:**

| product_id   | brand     | title                     | price  | category    | rating |
|--------------|----------|---------------------------|-------|------------|--------|
| B08YRWN3WB   | JANSPORT | Big Student Large Backpack| 189.0 | New season | 4.7    |
| B09Q2PQ7ZB   | BAODINI  | Mini Travel Umbrella      | 17.79 | New season | 4.2    |

---

## **Requirements**

- Python 3.10+
- TensorFlow 2.14+
- Pandas
- NumPy
- Matplotlib / Seaborn (optional, for plotting results)

**Install dependencies:**

```bash
pip install tensorflow pandas numpy matplotlib seaborn
```

## **Data Preprocessing**

**The following steps are applied to prepare the images and labels for training:**

| Step | Description |
|------|-------------|
| **Resize Images** | Images are resized to **(IMG_SIZE, IMG_SIZE)** to match the input shape of MobileNetV2. |
| **One-Hot Encoding** | Labels are **one-hot encoded** for multi-class classification. |
| **Data Augmentation** | Random transformations are applied using **ImageDataGenerator** to improve model generalization. |

**Data Augmentation Code Example:**

```bash
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_gen = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
```

## **Model Architecture**

**The model uses MobileNetV2 as a base model with frozen pretrained layers, followed by a custom classifier for multi-class fashion product classification.**

| Component            | Description |
|---------------------|-------------|
| **Base Model**       | MobileNetV2 pretrained on ImageNet with `include_top=False`. The weights are frozen (`trainable=False`). |
| **Flatten Layer**    | Converts feature maps into a 1D vector. |
| **Dense Layer**      | Fully connected layer with 256 neurons and ReLU activation. |
| **Dropout Layer**    | Dropout with rate 0.5 to reduce overfitting. |
| **Output Layer**     | Dense layer with `num_classes` neurons and softmax activation for multi-class classification. |
| **Optimizer & Loss** | Adam optimizer with learning rate 1e-3 and categorical crossentropy loss. |

**Python Code Example:**

```bash
import tensorflow as tf
```
```bash
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze pretrained layers
```

# Build model

```bash
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

# Compile model

```bash
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## **Training**

**The model is trained using EarlyStopping and ReduceLROnPlateau callbacks to improve performance and prevent overfitting.**

| Component                       | Description |
|--------------------------------|-------------|
| **EarlyStopping**               | Stops training when the validation loss does not improve for **5 epochs** and restores the best model weights. |
| **ReduceLROnPlateau**           | Reduces the learning rate by a factor of **0.5** if the validation loss does not improve for **3 epochs**. |
| **Epochs**                      | The model is trained for up to **25 epochs**. |
| **Validation Data**             | `(X_test, y_test)` is used to evaluate model performance after each epoch. |
| **Verbose**                      | Training progress is displayed for each epoch. |

**Python Code Example:**

```bash
import tensorflow as tf
```

# Define callbacks
```bash
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
```

# Train the model
```bash
history = model.fit(
    train_gen,
    epochs=25,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
```

## **Results**

**The model performance after training is summarized below:**

| Metric                   | Value/Observation |
|--------------------------|-----------------|
| **Training Accuracy**     | ~57–58%         |
| **Validation Accuracy**   | ~60–61%         |
| **Loss**                  | Stabilized around ~1.14–1.15 |
| **Observation**           | The model performs reasonably for a first iteration using MobileNetV2 with frozen layers. Fine-tuning the base model or using advanced data augmentation could further improve accuracy. |

---

## **Usage**

**Load the trained model:**

```bash
import tensorflow as tf

model = tf.keras.models.load_model("fashion_model.h5")
```

