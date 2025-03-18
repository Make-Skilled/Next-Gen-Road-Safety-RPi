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
frame_skip = 20  # Increased to process fewer frames
frame_count = 0
last_processing_time = 0
min_processing_interval = 0.3  # Increased interval between detections
detection_enabled = True
camera = None
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
detection_lock = threading.Lock()
latest_processed_frame = None
processed_frame_lock = threading.Lock()
detection_frame_buffer = None  # Buffer for detection frames

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
    """Initialize camera if not already initialized"""
    global camera
    try:
        if camera is not None:
            return True
            
        print("Initializing camera...")
        camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not camera.isOpened():
            print("Failed to open camera")
            return False
            
        # Optimize camera settings for performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Reduced resolution
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        camera.set(cv2.CAP_PROP_FPS, 10)  # Reduced FPS
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def release_camera():
    """Release camera resources"""
    global camera
    try:
        if camera is not None:
            print("Releasing camera...")
            camera.release()
            camera = None
            print("Camera released successfully")
    except Exception as e:
        print(f"Error releasing camera: {e}")

def camera_thread():
    """Thread for capturing camera frames"""
    global latest_frame, camera
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            if not init_camera():
                retry_count += 1
                if retry_count >= max_retries:
                    print("Failed to initialize camera after multiple attempts")
                    time.sleep(5)
                    retry_count = 0
                continue
            
            while camera is not None:
                ret, frame = camera.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Only flip and update frame, no other processing
                frame = cv2.flip(frame, 1)
                with frame_lock:
                    latest_frame = frame  # No need to copy for camera feed
                
                # Minimal sleep to prevent CPU overload while maintaining high FPS
                time.sleep(0.001)
        
        except Exception as e:
            print(f"Error in camera thread: {e}")
            release_camera()
            time.sleep(1)

def detection_thread():
    """Thread for object detection"""
    global latest_frame, latest_detections, frame_count, last_processing_time, latest_processed_frame, detection_frame_buffer
    
    while True:
        try:
            if not detection_enabled:
                latest_detections = []
                with processed_frame_lock:
                    latest_processed_frame = None
                time.sleep(0.2)
                continue
            
            current_time = time.time()
            if current_time - last_processing_time < min_processing_interval:
                time.sleep(0.1)  # Longer sleep when not processing
                continue
            
            # Get latest frame
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                frame = latest_frame.copy()
            
            try:
                # Resize frame for faster processing
                height, width = frame.shape[:2]
                frame_resized = cv2.resize(frame, (96, 96))  # Even smaller size for detection
                
                # Run detection with optimized parameters
                results = model(frame_resized, conf=confidence_threshold, verbose=False, half=True)[0]
                
                # Process detections
                detections = []
                processed_frame = frame.copy()
                
                # Process all detections in batch
                boxes = results.boxes
                if len(boxes) > 0:
                    # Get all coordinates, confidences, and class IDs at once
                    coords = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    
                    # Scale all coordinates at once
                    scale_x = width / 96
                    scale_y = height / 96
                    coords[:, [0, 2]] *= scale_x
                    coords[:, [1, 3]] *= scale_y
                    coords = coords.astype(int)
                    
                    # Draw all boxes at once for better performance
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = coords[i]
                        confidence = float(confs[i])
                        class_name = results.names[class_ids[i]]
                        
                        # Simplified drawing
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(processed_frame, class_name, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        detections.append({
                            'object_name': class_name,
                            'confidence': confidence,
                            'box': [x1, y1, x2 - x1, y2 - y1],
                            'timestamp': datetime.datetime.now()
                        })
                
                # Update detection results
                with detection_lock:
                    latest_detections = detections
                with processed_frame_lock:
                    latest_processed_frame = processed_frame
                    
                # Save to database very infrequently (every 3 seconds)
                if detections and time.time() - last_processing_time > 3.0:
                    with app.app_context():
                        for detection in detections:
                            new_detection = Detection(
                                object_name=detection['object_name'],
                                confidence=detection['confidence'],
                                user_id=current_user.id if current_user else None
                            )
                            db.session.add(new_detection)
                        db.session.commit()
                
                last_processing_time = current_time
                
            except Exception as e:
                print(f"Error processing detection: {e}")
            
            time.sleep(0.1)  # Longer delay between detections
            
        except Exception as e:
            print(f"Error in detection thread: {e}")
            time.sleep(0.2)

def gen_frames(user_id, feed_type='camera'):
    """Generate frames for video feed"""
    last_frame_time = 0
    frame_interval = 0.016 if feed_type == 'camera' else 0.1  # 60fps for camera, 10fps for detection
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Minimal sleep for camera feed
                continue
            
            # Get appropriate frame based on feed type
            if feed_type == 'detection':
                with processed_frame_lock:
                    if latest_processed_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = latest_processed_frame
                    # Lower quality for detection feed
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 65]
            else:
                with frame_lock:
                    if latest_frame is None:
                        time.sleep(0.001)
                        continue
                    frame = latest_frame
                    # Keep high quality for camera feed
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            last_frame_time = current_time
            
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
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug mode to prevent reloading 