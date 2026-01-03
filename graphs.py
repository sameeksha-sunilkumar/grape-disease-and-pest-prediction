import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

MODEL_PATH = "models/unified_leaf_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"random_brightness": random_brightness})
print("Model loaded successfully!")

VAL_DIR = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = val_ds.class_names  
print("Leaf Disease Classes:", class_names)

val_ds = val_ds.map(lambda x, y: (x / 255.0, y)) 

try:
    with open("history.pkl", "rb") as f:
        history = pickle.load(f)
    has_history = True
    print("Training history loaded!")
except FileNotFoundError:
    has_history = False
    print("Training history not found. Skipping accuracy/loss plots.")

if has_history:
    plt.figure(figsize=(10,4))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Epochs')
    plt.legend()
    plt.show()

y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n")
print(report)

for images, labels in val_ds.take(1):
    plt.figure(figsize=(12,8))
    for i in range(min(9, images.shape[0])):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy())
        pred_label = class_names[np.argmax(model.predict(tf.expand_dims(images[i],0)))]
        true_label = class_names[labels[i]]
        plt.title(f"T: {true_label} | P: {pred_label}")
        plt.axis('off')
    plt.show()
