import tensorflow as tf
from tensorflow.keras import layers, models
import os
from collections import Counter

train_dir = r"C:\Users\DELL\Desktop\projects\grape disease detection\fruit_data\train"
val_dir   = r"C:\Users\DELL\Desktop\projects\grape disease detection\fruit_data\test"

img_size = (160, 160)
batch_size = 32
epochs = 20
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Fruit Disease Classes:", class_names)

labels = []
for _, y in train_ds.unbatch():
    labels.append(int(y.numpy()))

label_counts = Counter(labels)
total = sum(label_counts.values())

class_weights = {
    cls: total / (len(label_counts) * count)
    for cls, count in label_counts.items()
}

print("Class Weights:")
for k, v in class_weights.items():
    print(f"{class_names[k]}: {v:.2f}")

train_ds = train_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)


model = models.Sequential([
    layers.Input(shape=(*img_size, 3)),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print("Training 3-Class Fruit Disease Model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stop]
)

os.makedirs("models", exist_ok=True)
model.save("models/fruit_3class_model.keras")
print("3-Class Fruit Disease Model Saved: models/fruit_3class_model.keras")
