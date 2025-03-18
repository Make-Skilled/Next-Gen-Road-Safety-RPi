import cv2
import numpy as np
import time
from datetime import datetime

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
                    'timestamp': datetime.now()
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