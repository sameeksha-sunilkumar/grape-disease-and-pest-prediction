import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

MODEL_PATH = "models/unified_leaf_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"random_brightness": random_brightness})
print(" Model loaded successfully!")

VAL_DIR = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\test"
TRAIN_DIR = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\train"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15  

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.show()

