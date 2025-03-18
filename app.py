from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import datetime
import os
from object_detection import ObjectDetector
import json
from simple_websocket import Server
import time
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables for camera handling
detector = None
camera_lock = threading.Lock()

def init_detector():
    global detector
    try:
        if detector is not None:
            detector.release()  # Release the camera before reinitializing
        detector = ObjectDetector()
        print("Detector initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing detector: {e}")
        if detector is not None:
            detector.release()
        detector = None
        return False

# Initialize detector
init_detector()

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

def gen_frames():
    global detector
    if detector is None:
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
    global detector
    if detector is None:
        if not init_detector():  # Try to initialize detector if not already initialized
            flash('Camera not available. Please check your camera connection.')
            return redirect(url_for('index'))
    detections = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.timestamp.desc()).all()
    return render_template('dashboard.html', detections=detections)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/detect')
@login_required
def detect():
    global detector
    try:
        if detector is None:
            init_detector()  # Try to reinitialize the detector
            if detector is None:
                flash('Camera not available. Please check your camera connection.')
                return redirect(url_for('dashboard'))
        return render_template('detect.html')
    except Exception as e:
        print(f"Error in detect route: {e}")
        flash('Error initializing camera. Please try again.')
        return redirect(url_for('dashboard'))

@app.route('/video_feed')
@login_required
def video_feed():
    global detector
    if detector is None:
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