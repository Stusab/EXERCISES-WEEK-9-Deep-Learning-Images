
# ✅ CHEAT SHEET – WEEK 9: Transfer Learning (Smile vs. No Smile)

## 🔍 1. Ziel
- Klassifiziere Bilder in **smiling** und **not_smiling**
- Nutze **MobileNetV2** als Pretrained Feature Extractor
- Wende **Data Augmentation** an
- Vergleiche:
  - 🔸 Training ohne Fine-Tuning
  - 🔹 Training mit Fine-Tuning

---

## 🧭 2. Daten vorbereiten (Genki4k Dataset)
```python
data_dir = "/content/drive/MyDrive/genki4k"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(192, 192),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(192, 192),
    batch_size=32
)
```

---

## 🌀 3. Data Augmentation
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
```

---

## 📦 4. Transfer Learning Setup (ohne Fine-Tuning)
```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(192, 192, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False
```

---

## 🛠 5. Modell aufbauen mit Functional API
```python
inputs = tf.keras.Input(shape=(192, 192, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
```

---

## ⚙️ 6. Kompilieren
```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

---

## 🏋️ 7. Training (Feature Extraction)
```python
initial_epochs = 20
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=initial_epochs
)
```

---

## 🔁 8. Fine-Tuning aktivieren
```python
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    initial_epoch=initial_epochs
)
```

---

## 📊 9. Ergebnisse plotten
```python
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']
```

---

## 📈 10. Interpretation
| Phase              | Effekt                                        |
|--------------------|-----------------------------------------------|
| Transfer Learning  | stabil, aber begrenzte Accuracy (~0.5–0.6)    |
| Fine-Tuning        | kann helfen, aber Gefahr von Overfitting      |

---

## 💡 Empfehlungen
- Fine-Tune nur 20–50 Layer
- Lernrate beim Fine-Tuning reduzieren
- Nutze Dropout & EarlyStopping
- Nur Fine-Tunen, wenn val_accuracy vorher stabil war
