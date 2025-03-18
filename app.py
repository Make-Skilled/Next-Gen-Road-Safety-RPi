from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import os
import json
import time
import threading
import atexit
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables for detection settings
confidence_threshold = 0.3
frame_skip = 12  # Process every 12th frame to reduce CPU load
frame_count = 0
last_processing_time = 0
min_processing_interval = 0.3  # Minimum time between processing frames (300ms)
detection_enabled = True  # Global flag to toggle detection
camera = None  # Global camera instance
latest_frame = None  # Latest frame from camera
latest_detections = []  # Latest detections
frame_lock = threading.Lock()  # Lock for thread-safe frame access
detection_lock = threading.Lock()  # Lock for thread-safe detection access

# Load YOLOv8 model
try:
    print("Loading YOLOv8 model...")
    # Try loading with different approaches
    try:
        # First try loading directly
        model = YOLO('yolov8n.pt')
    except Exception as e1:
        print(f"First attempt failed: {e1}")
        try:
            # Try downloading the model first
            from ultralytics import download
            print("Downloading model...")
            # Download the smallest model (nano)
            download(url='https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt', dir='.')
            model = YOLO('yolov8n.pt')
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            raise Exception("Failed to load YOLOv8 model. Please ensure you have enough disk space and internet connection.")
    
    # Optimize model for inference
    if hasattr(model, 'fuse'):
        model.fuse()  # Fuse Conv2d + BatchNorm2d layers
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    print("Please ensure you have internet connection and enough disk space")
    raise

# Load COCO labels
def load_labels():
    try:
        with open("coco_labels.txt", 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Warning: coco_labels.txt not found. Using default COCO labels.")
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']

labels = load_labels()

def check_system_resources():
    """Check if system has enough resources to process frame"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # Skip processing if CPU or memory usage is too high
    if cpu_percent > 90 or memory.percent > 90:
        print(f"System resources low - CPU: {cpu_percent}%, Memory: {memory.percent}%")
        return False
    return True

def init_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def camera_thread():
    """Thread for capturing camera frames"""
    global latest_frame, camera
    try:
        init_camera()
        while True:
            ret, frame = camera.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Flip for selfie view
                with frame_lock:
                    latest_frame = frame.copy()
            time.sleep(0.01)  # Small delay to prevent CPU overload
    except Exception as e:
        print(f"Error in camera thread: {e}")
    finally:
        release_camera()

def detection_thread():
    """Thread for object detection"""
    global latest_frame, latest_detections, frame_count, last_processing_time
    
    while True:
        try:
            if not detection_enabled:
                time.sleep(0.1)
                continue
                
            # Skip frames to reduce CPU load
            frame_count += 1
            if frame_count % frame_skip != 0:
                time.sleep(0.01)
                continue
                
            # Check if enough time has passed since last processing
            current_time = time.time()
            if current_time - last_processing_time < min_processing_interval:
                time.sleep(0.01)
                continue
            
            # Get latest frame
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_frame.copy()
            
            # Check system resources
            if not check_system_resources():
                time.sleep(0.1)
                continue
            
            # Process frame
            height, width = frame.shape[:2]
            frame_resized = cv2.resize(frame, (192, 192))
            
            # Run detection
            results = model(frame_resized, conf=confidence_threshold, verbose=False, half=True)[0]
            
            # Process detections
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                # Scale coordinates
                x1 = int(x1 * width / 192)
                y1 = int(y1 * height / 192)
                x2 = int(x2 * width / 192)
                y2 = int(y2 * height / 192)
                
                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                detections.append({
                    'object_name': class_name,
                    'confidence': confidence,
                    'box': [x1, y1, x2 - x1, y2 - y1],
                    'timestamp': datetime.datetime.now()
                })
            
            # Update latest detections
            with detection_lock:
                latest_detections = detections
            
            last_processing_time = current_time
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in detection thread: {e}")
            time.sleep(0.1)

def gen_frames(user_id, feed_type='camera'):
    """Generate frames for video feed"""
    with app.app_context():
        while True:
            try:
                # Get latest frame
                with frame_lock:
                    if latest_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = latest_frame.copy()
                
                # For detection feed, add bounding boxes
                if feed_type == 'detection':
                    with detection_lock:
                        detections = latest_detections.copy()
                    
                    # Draw detections on frame
                    for detection in detections:
                        x1, y1, w, h = detection['box']
                        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                        label = f"{detection['object_name']}: {detection['confidence']:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Save detections to database
                    if user_id and detections:
                        for detection in detections:
                            new_detection = Detection(
                                object_name=detection['object_name'],
                                confidence=detection['confidence'],
                                user_id=user_id
                            )
                            db.session.add(new_detection)
                        db.session.commit()
                
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 85,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_SAMPLING_FACTOR, 422
                ])
                
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in gen_frames: {e}")
                time.sleep(0.1)

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
    feed_type = request.args.get('type', 'camera')
    return Response(gen_frames(current_user.id, feed_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_threshold', methods=['POST'])
@login_required
def update_threshold():
    global confidence_threshold
    data = request.get_json()
    confidence_threshold = float(data.get('threshold', 0.3))
    return jsonify({'status': 'success'})

@app.route('/toggle_detection', methods=['POST'])
@login_required
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({'status': 'success', 'detection_enabled': detection_enabled})

# Register cleanup function
atexit.register(release_camera)

# Start threads when the application starts
def start_threads():
    camera_thread_instance = threading.Thread(target=camera_thread, daemon=True)
    detection_thread_instance = threading.Thread(target=detection_thread, daemon=True)
    camera_thread_instance.start()
    detection_thread_instance.start()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    start_threads()  # Start the threads
    app.run(host='0.0.0.0', port=5000, debug=True) 