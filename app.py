from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import datetime
import os
import json
from simple_websocket import Server
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
detector = None
camera_lock = threading.Lock()
camera_initialized = False

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        # Load COCO class labels
        self.labels_path = "coco_labels.txt"
        self.labels = self._load_labels()
        
        # Load YOLO model
        self.config_path = "yolov3-tiny.cfg"
        self.weights_path = "yolov3-tiny.weights"
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        
        # Set backend and target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.3
        
        # Initialize camera
        self.camera = None
        self._init_camera()
        
    def _init_camera(self):
        try:
            # First, try to release any existing camera
            if self.camera is not None:
                self.camera.release()
                time.sleep(1)  # Wait for camera to fully release
                
            # Try to open the camera
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L)
            if not self.camera.isOpened():
                print("Failed to open camera, trying alternative method...")
                self.camera = cv2.VideoCapture(0)  # Try without CAP_V4L
                if not self.camera.isOpened():
                    raise Exception("Failed to open camera with both methods")
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
            
            # Wait for camera to initialize
            time.sleep(2)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Could not read from camera")
                
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            if self.camera is not None:
                self.camera.release()
            self.camera = None
            return False
            
    def _load_labels(self):
        try:
            with open(self.labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: {self.labels_path} not found. Using default COCO labels.")
            return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
    
    def detect_objects(self, frame):
        if frame is None:
            return None, []
            
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.net.forward(output_layers)
        
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
                
                if confidence > self.confidence_threshold:
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
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detection = {
                    'object_name': self.labels[class_ids[i]],
                    'confidence': confidences[i],
                    'box': boxes[i],
                    'timestamp': datetime.datetime.now()
                }
                detections.append(detection)
                
                # Draw bounding box and label on frame
                (x, y, w, h) = boxes[i]
                label = f"{self.labels[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, detections
    
    def get_frame(self):
        try:
            # Check if camera needs reinitialization
            if self.camera is None or not self.camera.isOpened():
                print("Camera not available, attempting to reinitialize...")
                if not self._init_camera():
                    print("Failed to reinitialize camera")
                    return None, []
                    
            # Try to read frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame, attempting to reinitialize camera...")
                if not self._init_camera():
                    print("Failed to reinitialize camera after frame grab failure")
                    return None, []
                ret, frame = self.camera.read()
                if not ret:
                    print("Still failed to grab frame after reinitialization")
                    return None, []
                    
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Detect objects
            frame, detections = self.detect_objects(frame)
            return frame, detections
            
        except Exception as e:
            print(f"Error in get_frame: {e}")
            return None, []
    
    def release(self):
        try:
            if self.camera is not None and self.camera.isOpened():
                self.camera.release()
                self.camera = None
                print("Camera released successfully")
        except Exception as e:
            print(f"Error releasing camera: {e}")

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

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

def init_detector():
    global detector, camera_initialized
    try:
        if detector is not None:
            detector.release()  # Release the camera before reinitializing
            time.sleep(1)  # Wait for camera to fully release
            
        detector = ObjectDetector()
        if detector.camera is None:
            raise Exception("Failed to initialize camera")
            
        print("Detector initialized successfully")
        camera_initialized = True
        return True
    except Exception as e:
        print(f"Error initializing detector: {e}")
        if detector is not None:
            detector.release()
        detector = None
        camera_initialized = False
        return False

def cleanup_detector():
    global detector, camera_initialized
    if detector is not None:
        detector.release()
        detector = None
        camera_initialized = False
        print("Detector cleaned up")

# Initialize detector at startup
init_detector()

# Register cleanup function
atexit.register(cleanup_detector)

def gen_frames():
    global detector, camera_initialized
    if not camera_initialized or detector is None:
        print("Detector not initialized")
        return
        
    while True:
        try:
            with camera_lock:
                frame, detections = detector.get_frame()
                
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
                print("No frame received")
                time.sleep(0.1)  # Small delay before retrying
        except Exception as e:
            print(f"Error in gen_frames: {e}")
            time.sleep(1)  # Longer delay on error

@app.route('/')
def index():
    if detector is None:
        init_detector()  # Try to initialize detector if not already initialized
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
    global detector, camera_initialized
    if not camera_initialized or detector is None:
        if not init_detector():
            flash('Camera not available. Please check your camera connection.')
            return redirect(url_for('index'))
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
    global detector, camera_initialized
    if not camera_initialized or detector is None:
        if not init_detector():
            return "Camera not available", 503
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_threshold', methods=['POST'])
@login_required
def update_threshold():
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Camera not available'}), 503
    data = request.get_json()
    threshold = float(data.get('threshold', 0.5))
    detector.set_confidence_threshold(threshold)
    return jsonify({'status': 'success'})

@app.route('/ws/detection')
@login_required
def ws_detection():
    if detector is None:
        return "Camera not available", 503
    ws = Server(request.environ)
    try:
        while True:
            with camera_lock:
                _, detections = detector.get_frame()
            if detections:
                # Send only the latest detection
                latest = detections[-1]
                ws.send(json.dumps({
                    'object_name': latest['object_name'],
                    'confidence': latest['confidence'],
                    'timestamp': latest['timestamp'].isoformat()
                }))
    except:
        ws.close()
    return ''

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True) 