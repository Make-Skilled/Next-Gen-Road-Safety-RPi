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

# Global variables for detection settings
confidence_threshold = 0.3
nms_threshold = 0.2

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

def detect_objects(frame):
    if frame is None:
        print("Frame is None in detect_objects")
        return None, []
        
    height, width = frame.shape[:2]
    print(f"Processing frame of size: {width}x{height}")
    
    try:
        # Preprocess image for better detection
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Apply slight blur to reduce noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Create blob from image with adjusted parameters
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        print("Created blob successfully")
        
        net.setInput(blob)
        print("Set input to network")
        
        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        print(f"Output layers: {output_layers}")
        
        # Forward pass
        outputs = net.forward(output_layers)
        print(f"Forward pass completed. Number of outputs: {len(outputs)}")
        
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
                
                # Print top 3 classes and their confidences for debugging
                top_3_indices = np.argsort(scores)[-3:][::-1]
                print(f"Top 3 classes: {[labels[i] for i in top_3_indices]}")
                print(f"Top 3 confidences: {[scores[i] for i in top_3_indices]}")
                
                if confidence > confidence_threshold:
                    # Scale bounding box coordinates back relative to size of image
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # Use center coordinates to derive top and left corner of bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, min(x, width))
                    y = max(0, min(y, height))
                    width = min(width, frame.shape[1] - x)
                    height = min(height, frame.shape[0] - y)
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        print(f"Found {len(boxes)} potential detections")
        
        # Apply non-maxima suppression with adjusted threshold
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        print(f"After NMS: {len(indices)} detections")
        
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
                
                # Draw thicker bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Add background to text for better visibility
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame, detections
        
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        return frame, []

def get_camera_frame():
    try:
        # Create a new camera instance for each frame
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Failed to open camera")
            return None, []
            
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Read frame
        ret, frame = camera.read()
        camera.release()  # Release camera immediately after reading
        
        if not ret:
            print("Failed to read frame")
            return None, []
            
        print("Frame captured successfully")
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Detect objects
        frame, detections = detect_objects(frame)
        print(f"Number of detections: {len(detections)}")
        return frame, detections
        
    except Exception as e:
        print(f"Error in get_camera_frame: {e}")
        return None, []

def gen_frames(user_id):
    while True:
        try:
            frame, detections = get_camera_frame()
            
            if frame is not None:
                # Save detections to database if user_id is provided
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
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                print("No frame received, waiting...")
                time.sleep(0.5)
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
    return Response(gen_frames(current_user.id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_threshold', methods=['POST'])
@login_required
def update_threshold():
    global confidence_threshold
    data = request.get_json()
    confidence_threshold = float(data.get('threshold', 0.3))
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True) 