import numpy as np
import cv2
import csv
import face_recognition
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
from flask import Flask, render_template, redirect, request, flash, url_for, session, jsonify,Response
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from io import BytesIO
app = Flask(__name__)
SECRET_KEY = "Momina0192004." 
# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Momina0192004.'  # Update with your actual password
app.config['MYSQL_DB'] = 'face_attendance'
app.config['SESSION_COOKIE_NAME'] = 'admin_session'
app.config['SECRET_KEY'] = SECRET_KEY  # Add this line

mysql = MySQL(app)  # Initialize MySQL object with Flask app

# Load the pre-trained face detector and mask detector models
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
mask_net = load_model("mask_detector.model.h5")

# Database connection function (using Flask MySQLdb)
def get_db_connection():
    try:
        conn = mysql.connect  # Use the `mysql.connect` method from Flask MySQLdb
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Login required decorator
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return wrapper

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    return render_template('admin_dashboard.html')


@app.route('/attendance_sheet', methods=['GET'])
def attendance_sheet_page():
    
    
    # Fetch attendance records
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT date_time,attendance_count,username,registration_id FROM attendance")
    attendance_records = cursor.fetchall()
    conn.close()

    return render_template('attendance_sheet.html', attendance=attendance_records)


# Route to download attendance CSV
@app.route('/admin/download_attendance', methods=['GET'])
def download_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT date_time,attendance_count,username,registration_id FROM attendance")
    attendance_records = cursor.fetchall()

    def generate():
        yield 'Date Time,Attendance Count,Username,Registration ID\n'
        for record in attendance_records:
            yield f"{record[0]},{record[1]},{record[2]},{record[3]}\n"

    return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=attendance.csv"})
# Detect Faces function (no changes)
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

def capture_face(mode, cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to capture frame."}), 500

        locs, preds = detect_and_predict_mask(frame)
        label = "No Mask" if mode == "bare" else "Mask"
        valid_capture = False

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            current_label = "Mask" if mask > withoutMask else "No Mask"

            if current_label == label:
                valid_capture = True
                face_locations = detect_faces(frame)
                if face_locations:
                    encodings = face_recognition.face_encodings(frame, face_locations)
                    if encodings:
                        cv2.putText(frame, f"Captured {label} face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(f"Register {mode.capitalize()} Face", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return encodings[0]  # Only capture when the correct face is detected

        if not valid_capture:
            instruction = f"Please {'remove' if mode == 'bare' else 'wear'} your mask to proceed."
            cv2.putText(frame, instruction, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow(f"Register {mode.capitalize()} Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') and valid_capture:
            break

    return None

@app.route('/register', methods=['GET', 'POST'])
def register_user():
    if request.method == 'GET':
        return render_template('register.html')

    if request.method == 'POST':
        name = request.form.get("name")
        registration_id = request.form.get("registration_id")

        if not name or not registration_id:
            return jsonify({"error": "Name and Registration ID cannot be empty."}), 400

        cap = cv2.VideoCapture(0)

        bare_face_encoding = capture_face("bare", cap)
        if bare_face_encoding is None:
            return jsonify({"error": "Failed to capture bare face."}), 500

        print("Please wear a mask and capture your masked face.")
        masked_face_encoding = capture_face("masked", cap)
        if masked_face_encoding is None:
            return jsonify({"error": "Failed to capture masked face."}), 500

        # Ensure the face encodings are valid arrays
        if bare_face_encoding is None or masked_face_encoding is None:
            return jsonify({"error": "Invalid face encodings captured."}), 500

        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    # Check if the user already exists by registration ID
                    cursor.execute("SELECT * FROM students WHERE registration_id = %s", (registration_id,))
                    existing_user = cursor.fetchone()
                    if existing_user:
                        return jsonify({"error": "This registration ID is already registered."}), 400

                    cursor.execute(
                        "INSERT INTO students (registration_id, name, bare_face_encoding, masked_face_encoding) VALUES (%s, %s, %s, %s)",
                        (registration_id, name, bare_face_encoding.tobytes(), masked_face_encoding.tobytes())
                    )
                    conn.commit()
                print(f"User {name} registered successfully.")
                return jsonify({"success": f"User '{name}' registered successfully."}), 200
            except mysql.connector.Error as err:
                print(f"Error: {err}")
                return jsonify({"error": "Failed to save user data."}), 500
            finally:
                conn.close()
        else:
            return jsonify({"error": "Failed to connect to the database."}), 500

@app.route('/mark_attendance', methods=['GET'])
def mark_attendance_page():
    return render_template('mark_attendance.html')
@app.route('/mark_attendance', methods=['POST'])
def start_attendance_marking():
    # Check if 'image' is in the request files
    if 'image' not in request.files:
        return jsonify({"error": "No image found in request."}), 400

    # Get the image file from the request
    image_file = request.files['image']
    if not image_file:
        return jsonify({"error": "No image found in request."}), 400

    # Read the image file into a NumPy array
    nparr = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame to detect and match faces (you can add your face recognition logic here)
    attendance_marked = False
    unregistered_user = False

    locs, preds = detect_and_predict_mask(frame)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        current_label = "Mask" if mask > withoutMask else "No Mask"

        face_locations = detect_faces(frame)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_encoding = face_recognition.face_encodings(frame, [face_location])

            if face_encoding:
                face_encoding = face_encoding[0]

                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, registration_id, name, bare_face_encoding, masked_face_encoding FROM students")
                    users = cursor.fetchall()

                    matched_user = None
                    for user in users:
                        user_id, registration_id, username, bare_face_encoding, masked_face_encoding = user
                        bare_face_encoding = np.frombuffer(bare_face_encoding, dtype=np.float64)
                        masked_face_encoding = np.frombuffer(masked_face_encoding, dtype=np.float64)

                        match_bare = face_recognition.compare_faces([bare_face_encoding], face_encoding)
                        match_masked = face_recognition.compare_faces([masked_face_encoding], face_encoding)

                        if match_bare[0] or match_masked[0]:
                            matched_user = (user_id, registration_id, username)
                            break

                    if matched_user:
                        user_id, registration_id, username = matched_user
                        
                        # Check if attendance for the current date has already been marked for this user
                        cursor.execute("""
                            SELECT COUNT(*) 
                            FROM attendance 
                            WHERE user_id = %s AND DATE(date_time) = %s
                        """, (user_id, datetime.now().date()))

                        attendance_count = cursor.fetchone()[0]  # Fetch the count of today's attendance

                        # If attendance has not been marked for today, insert the attendance record
                        if attendance_count == 0:
                            cursor.execute("""
                                INSERT INTO attendance (user_id, registration_id, username, date_time, attendance_count)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (user_id, registration_id, username, datetime.now(), 1))  # Set attendance_count as 1
                            conn.commit()

                            print(f"Attendance marked for {username}.")
                            attendance_marked = True
                            return jsonify({"success": f"Attendance marked for {username}."}), 200
                        else:
                            return jsonify({"info": f"Attendance for {username} already marked today."}), 200
                    else:
                        return jsonify({"error": "Unregistered user detected!"}), 400

    if attendance_marked:
        return jsonify({"success": "Attendance successfully marked."}), 200
    else:
        return jsonify({"error": "Attendance marking failed."}), 500


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            cursor = mysql.connection.cursor()  # Correct method for cursor
            cursor.execute("SELECT id, first_name, last_name, email, password_hash FROM admin_users WHERE email = %s", (email,))
            user = cursor.fetchone()
            cursor.close()

            if user:
                if check_password_hash(user[4], password):
                    session['logged_in'] = True
                    session['admin_id'] = user[0]  # Save admin ID in session
                    return jsonify({'success': True, 'admin_id': user[0]})  # Return admin ID for localStorage
                else:
                    flash("Invalid email or password", "danger")
            else:
                flash("Invalid email or password", "danger")

            return redirect(url_for('login'))

        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        contact = request.form.get('contact')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        secret_key = request.form.get('secret_key')

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('signup'))

        if secret_key != SECRET_KEY:
            flash("Invalid secret key.", "danger")
            return redirect(url_for('signup'))

        password_hash = generate_password_hash(password)

        try:
            cursor = mysql.connection.cursor()
            cursor.execute(
                """
                INSERT INTO admin_users (first_name, last_name, email, contact, password_hash, secret_key)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (first_name, last_name, email, contact, password_hash, secret_key)
            )
            mysql.connection.commit()
            cursor.close()
            flash("User registered successfully! Please log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash("An error occurred during registration. Please try again.", "danger")
            print(f"Error: {e}")

    return render_template('signup.html')




# Logout route
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))
# Route to view all students
@app.route('/admin/students')
def view_students():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM students")
    students = cursor.fetchall()
    return render_template('students.html', students=students)

# Route to delete a student
@app.route('/admin/delete_student/<int:id>', methods=['GET'])
def delete_student(id):
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM students WHERE id = %s", [id])
    mysql.connection.commit()
    return redirect(url_for('view_students'))

# Route to update student information
@app.route('/admin/update_student/<int:id>', methods=['GET', 'POST'])
def update_student(id):
    cursor = mysql.connection.cursor()
    
    if request.method == 'GET':
        cursor.execute("SELECT * FROM students WHERE id = %s", [id])
        student = cursor.fetchone()
        return render_template('update_student.html', student=student)

    if request.method == 'POST':
        name = request.form['name']
        registration_id = request.form['registration_id']
        
        cursor.execute("UPDATE students SET name = %s, registration_id = %s WHERE id = %s", 
                       [name, registration_id, id])
        mysql.connection.commit()
        return redirect(url_for('view_students'))

if __name__ == '__main__':
    app.run(debug=True)
