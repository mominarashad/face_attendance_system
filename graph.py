import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Initialize hyperparameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"D:\5th Semester\AI\lasttry\data"
CATEGORIES = ["with_mask", "without_mask"]

# Load and preprocess data
print("[INFO] Loading images...")
data, labels = [], []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    if not os.path.exists(path):
        print(f"[ERROR] Directory not found: {path}")
        continue
    print(f"[INFO] Checking images in directory: {path}")
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"[ERROR] Failed to load image {img_path}: {e}")

print(f"[INFO] Loaded {len(data)} images.")

# Convert to arrays
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split dataset
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Class distribution
print("[INFO] Checking class distribution...")
class_counts = {CATEGORIES[i]: labels[:, i].sum() for i in range(len(CATEGORIES))}
plt.bar(class_counts.keys(), class_counts.values(), color=["blue", "red"])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# Handle class imbalance with SMOTE
trainX_flat = trainX.reshape(trainX.shape[0], -1)
smote = SMOTE(random_state=42)
trainX_resampled, trainY_resampled = smote.fit_resample(trainX_flat, trainY.argmax(axis=1))
trainX_resampled = trainX_resampled.reshape(trainX_resampled.shape[0], 224, 224, 3)
trainY_resampled = to_categorical(trainY_resampled)

print(f"[INFO] Applied SMOTE. New training shape: {trainX_resampled.shape}")

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Build model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("[INFO] Training model...")
H = model.fit(
    aug.flow(trainX_resampled, trainY_resampled, batch_size=BS),
    steps_per_epoch=len(trainX_resampled) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Evaluate model
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Confusion matrix
print("[INFO] Generating confusion matrix...")
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot training performance
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# Save model
print("[INFO] Saving model...")
model.save("mask_detector.model")
