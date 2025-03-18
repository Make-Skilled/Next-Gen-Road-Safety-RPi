from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import datetime
import os
import json
import time
import threading
import atexit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables for camera handling
camera = None
camera_lock = threading.Lock()
camera_initialized = False
confidence_threshold = 0.5
nms_threshold = 0.3

# Load YOLO model
config_path = "yolov3-tiny.cfg"
weights_path = "yolov3-tiny.weights"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO labels
def load_labels():
    try:
        with open("coco_labels.txt", 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Warning: coco_labels.txt not found. Using default COCO labels.")
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']

labels = load_labels()

def init_camera():
    global camera, camera_initialized
    try:
        # First, ensure any existing camera is properly released
        if camera is not None:
            camera.release()
            time.sleep(1)
        
        # Try to open camera with basic settings
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Failed to open camera")
            return False
            
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
        
        # Test camera with multiple attempts
        for _ in range(3):  # Try 3 times
            ret, frame = camera.read()
            if ret:
                print("Camera initialized successfully")
                camera_initialized = True
                return True
            time.sleep(0.1)  # Wait a bit between attempts
            
        print("Failed to read from camera after multiple attempts")
        camera.release()
        return False
        
    except Exception as e:
        print(f"Error initializing camera: {e}")
        if camera is not None:
            camera.release()
        camera = None
        camera_initialized = False
        return False

def detect_objects(frame):
    if frame is None:
        return None, []
        
    height, width = frame.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass
    outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Scale bounding box coordinates back relative to size of image
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, width, height) = box.astype("int")
                
                # Use center coordinates to derive top and left corner of bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detection = {
                'object_name': labels[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i],
                'timestamp': datetime.datetime.now()
            }
            detections.append(detection)
            
            # Draw bounding box and label on frame
            (x, y, w, h) = boxes[i]
            label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, detections

def get_frame():
    global camera, camera_initialized
    try:
        # Check if camera needs reinitialization
        if camera is None or not camera.isOpened():
            print("Camera not available, attempting to initialize...")
            if not init_camera():
                print("Failed to initialize camera")
                return None, []
        
        # Try to read frame with multiple attempts
        for _ in range(3):
            ret, frame = camera.read()
            if ret:
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Detect objects
                frame, detections = detect_objects(frame)
                return frame, detections
            time.sleep(0.1)
        
        print("Failed to read frame after multiple attempts")
        return None, []
        
    except Exception as e:
        print(f"Error in get_frame: {e}")
        return None, []

def release_camera():
    global camera, camera_initialized
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        camera_initialized = False
        print("Camera released successfully")

# Initialize camera at startup
init_camera()

# Register cleanup function
atexit.register(release_camera)

def gen_frames():
    while True:
        try:
            with camera_lock:
                frame, detections = get_frame()
                
            if frame is not None:
                # Save detections to database if user is logged in
                if current_user.is_authenticated and detections:
                    for detection in detections:
                        new_detection = Detection(
                            object_name=detection['object_name'],
                            confidence=detection['confidence'],
                            user_id=current_user.id
                        )
                        db.session.add(new_detection)
                    db.session.commit()

                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                print("No frame received, waiting...")
                time.sleep(0.5)  # Increased wait time
        except Exception as e:
            print(f"Error in gen_frames: {e}")
            time.sleep(1)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    detections = db.relationship('Detection', backref='user', lazy=True)

# Detection Model
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    object_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
            
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    detections = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.timestamp.desc()).all()
    return render_template('dashboard.html', detections=detections)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/video_feed')
@login_required
def video_feed():
    global camera_initialized
    if not camera_initialized:
        print("Camera not initialized, attempting to initialize...")
        if not init_camera():
            print("Failed to initialize camera")
            return "Camera not available", 503
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_threshold', methods=['POST'])
@login_required
def update_threshold():
    global confidence_threshold
    data = request.get_json()
    confidence_threshold = float(data.get('threshold', 0.5))
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True) 