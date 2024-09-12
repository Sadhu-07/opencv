import cv2
from ultralytics import YOLO
from mtcnn import MTCNN

# Initialize YOLOv8 model for object detection
yolo_model = YOLO('yolov8n.pt')  # Ensure the correct path and model file

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Function to preprocess image for better visibility in low light
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_eq = cv2.equalizeHist(gray)
    bgr_eq = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
    return bgr_eq

# Function to detect objects using YOLOv8
def detect_objects_yolo(image):
    results = yolo_model(image)  # Perform inference
    detections = []

    # Assuming results is a list of detection results
    for result in results:
        for detection in result.boxes:  # Access detection boxes
            x1, y1, x2, y2 = detection.xyxy[0]  # Coordinates of the bounding box
            conf = detection.conf[0]  # Confidence score
            cls = int(detection.cls[0])  # Class ID
            detections.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'confidence': conf, 'class': cls})

    return detections

# Function to detect faces using MTCNN
def detect_faces_mtcnn(image):
    faces = mtcnn.detect_faces(image)
    detections = []
    for face in faces:
        x, y, w, h = face['box']
        detections.append({'x': x, 'y': y, 'w': w, 'h': h})
    return detections

# Open webcam
cap = cv2.VideoCapture(0)

# Create a named window
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image
    preprocessed_frame = preprocess_image(frame)

    # Detect objects using YOLO
    yolo_detections = detect_objects_yolo(preprocessed_frame)
    print("YOLO Detections:", yolo_detections)  # Debug print

    # Detect faces using MTCNN
    mtcnn_detections = detect_faces_mtcnn(preprocessed_frame)
    print("MTCNN Detections:", mtcnn_detections)  # Debug print

    # Draw YOLO detections
    for det in yolo_detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        if det['confidence'] > 0.5:  # Confidence threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw MTCNN detections
    for face in mtcnn_detections:
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
