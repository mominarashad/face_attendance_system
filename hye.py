import mysql.connector
import tkinter as tk
from tkinter import simpledialog, messagebox
from mysql.connector import Error
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import face_recognition

# Load the pre-trained face detector and mask detector models
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
mask_net = load_model("mask_detector.model.h5")

# Database connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Momina0192004.',
            database='attendance_system',
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return None

# Detect faces using OpenCV DNN
def detect_faces(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    h, w = image.shape[:2]
    face_locations = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_locations.append((startY, endX, endY, startX))  # Convert to face_recognition format

    return face_locations

# Detect and predict mask status
def detect_and_predict_mask(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    if detections.shape[2] > 0:  # Ensure detections are non-empty
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = mask_net.predict(faces, batch_size=32)

    return locs, preds

# Unified registration function
def register_user():
    name = simpledialog.askstring("Input", "Please enter your name:")
    if not name:
        messagebox.showerror("Error", "Name cannot be empty.")
        return

    cap = cv2.VideoCapture(0)

    def capture_face(mode):
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            locs, preds = detect_and_predict_mask(frame)
            label = "No Mask" if mode == "bare" else "Mask"
            valid_capture = False

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                current_label = "Mask" if mask > withoutMask else "No Mask"

                # Check if the current label matches the required mode
                if current_label == label:
                    valid_capture = True
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, f"{current_label} - Press 'q' to capture", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                    # Encode the face once mask requirement is met
                    face_locations = detect_faces(frame)
                    if face_locations:
                        encodings = face_recognition.face_encodings(frame, face_locations)
                        if encodings:
                            return encodings[0]

            if not valid_capture:
                instruction = f"Please {'remove' if mode == 'bare' else 'wear'} your mask to proceed."
                cv2.putText(frame, instruction, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow(f"Register {mode.capitalize()} Face - Press 'q' to capture", frame)

            # Wait for the user to press 'q' only if the mask condition is satisfied
            if cv2.waitKey(1) & 0xFF == ord('q') and valid_capture:
                break

        return None

    print("Capture a bare-faced image. Press 'q' to capture.")
    bare_face_encoding = capture_face("bare")

    print("Now wear a mask and capture a masked face. Press 'q' to capture.")
    messagebox.showinfo("Next Step", "Please wear a mask and capture your masked face.")
    masked_face_encoding = capture_face("masked")

    cap.release()
    cv2.destroyAllWindows()

    if bare_face_encoding is not None and masked_face_encoding is not None:
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO users (name, bare_face_encoding, masked_face_encoding) VALUES (%s, %s, %s)",
                        (name, bare_face_encoding.tobytes(), masked_face_encoding.tobytes())
                    )
                    conn.commit()
                print(f"User {name} registered successfully.")
                messagebox.showinfo("Success", f"User '{name}' registered successfully.")
            except mysql.connector.Error as err:
                print(f"Error: {err}")
                messagebox.showerror("Database Error", "Failed to save user data.")
            finally:
                conn.close()

# GUI setup
root = tk.Tk()
root.title("Unified Face Recognition System")

register_btn = tk.Button(root, text="Register", command=register_user)
register_btn.pack(pady=10)

root.mainloop()
