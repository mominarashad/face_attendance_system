import mysql.connector
import tkinter as tk
from mysql.connector import Error
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from tkinter import simpledialog, messagebox

# Load the pre-trained DNN model
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Database connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Momina0192004.',
            database='attendance_system',  # Ensure database name matches your setup
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return None

# Detect face using OpenCV DNN
def detect_face(image):
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

# Register user
def register_user():
    name = simpledialog.askstring("Input", "Please enter your name:")
    if not name:
        messagebox.showerror("Error", "Name cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    print("Capture a bare-faced image. Press 'q' to capture.")

    bare_face_encoding = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        cv2.imshow("Register Bare Face - Press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_locations = detect_face(frame)
            if face_locations:
                encodings = face_recognition.face_encodings(frame, face_locations)
                if encodings:
                    bare_face_encoding = encodings[0]
                    break
            else:
                print("No face detected. Try again.")

    print("Now wear a mask and capture a masked face. Press 'q' to capture.")
    messagebox.showinfo("Next Step", "Please wear a mask and capture your masked face.")

    masked_face_encoding = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        cv2.imshow("Register Masked Face - Press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_locations = detect_face(frame)
            if face_locations:
                encodings = face_recognition.face_encodings(frame, face_locations)
                if encodings:
                    masked_face_encoding = encodings[0]
                    break
            else:
                print("No face detected. Try again.")

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

def mark_attendance():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to capture the face for attendance.")

    conn = get_db_connection()
    if not conn:
        return

    known_encodings = []
    known_names = []

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name, bare_face_encoding, masked_face_encoding FROM users")
            for row in cursor.fetchall():
                user_id, name, bare_face_encoding, masked_face_encoding = row
                if bare_face_encoding and masked_face_encoding:
                    bare_face = np.frombuffer(bare_face_encoding, dtype=np.float64)
                    masked_face = np.frombuffer(masked_face_encoding, dtype=np.float64)
                    known_encodings.append((bare_face, masked_face))
                    known_names.append((user_id, name))
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        messagebox.showerror("Database Error", "Failed to fetch user data.")
        return
    finally:
        conn.close()

    def find_match(face_encoding):
        for (bare_face, masked_face), (user_id, name) in zip(known_encodings, known_names):
            bare_match = face_recognition.compare_faces([bare_face], face_encoding, tolerance=0.5)[0]
            masked_match = face_recognition.compare_faces([masked_face], face_encoding, tolerance=0.5)[0]
            if bare_match or masked_match:
                return user_id, name
        return None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        cv2.imshow("Mark Attendance - Press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_locations = detect_face(frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                if face_encodings:
                    user_id, user_name = find_match(face_encodings[0])
                    if user_id:
                        conn = get_db_connection()
                        if conn:
                            try:
                                with conn.cursor() as cursor:
                                    current_date = datetime.now().strftime('%Y-%m-%d')
                                    cursor.execute(
                                        "SELECT * FROM attendance WHERE user_id = %s AND DATE(date_time) = %s",
                                        (user_id, current_date)
                                    )
                                    if cursor.fetchone():
                                        cursor.execute(
                                            "UPDATE attendance SET attendance_count = attendance_count + 1 "
                                            "WHERE user_id = %s AND DATE(date_time) = %s",
                                            (user_id, current_date)
                                        )
                                    else:
                                        cursor.execute(
                                            "INSERT INTO attendance (user_id, date_time, attendance_count, username) "
                                            "VALUES (%s, NOW(), %s, %s)",
                                            (user_id, 1, user_name)
                                        )
                                    conn.commit()
                                messagebox.showinfo("Success", f"Attendance marked for {user_name}.")
                            except mysql.connector.Error as err:
                                print(f"Error: {err}")
                            finally:
                                conn.close()
                    else:
                        print("No match found. Logging face data for review.")
                        messagebox.showwarning("No Match", "Face not recognized.")
                else:
                    messagebox.showerror("Error", "No face detected.")
            else:
                messagebox.showerror("Error", "No faces detected.")
            break

    cap.release()
    cv2.destroyAllWindows()


# GUI setup
root = tk.Tk()
root.title("Face Recognition Attendance System")

register_btn = tk.Button(root, text="Register", command=register_user)
register_btn.pack(pady=10)

attendance_btn = tk.Button(root, text="Mark Attendance", command=mark_attendance)
attendance_btn.pack(pady=10)

root.mainloop()
