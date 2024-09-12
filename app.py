import time
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO  # Assuming you are using YOLOv8

# Load Haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLOv8 model
yolo_model = YOLO("yolov8s.pt")

# Function to predict age and gender using multiple models
def predict_age_gender(image):
    predictions = []
    models = ['VGG-Face', 'Facenet']  # List of models to be used

    for model_name in models:
        try:
            # Perform the analysis using the DeepFace analyze function
            analysis = DeepFace.analyze(img_path=image, actions=['age', 'gender'], model_name=model_name, enforce_detection=False)
            predictions.append(analysis)
        except Exception as e:
            print(f"Error during DeepFace analysis with {model_name}: {e}")

    # Combine predictions (e.g., using averaging or majority voting)
    if predictions:
        combined_age = sum([pred['age'] for pred in predictions]) / len(predictions)
        gender_votes = [pred['gender'] for pred in predictions]
        combined_gender = max(set(gender_votes), key=gender_votes.count)  # Majority vote for gender

        return combined_age, combined_gender
    return None, None

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error opening webcam")
    exit()

# Set up a figure for matplotlib
plt.ion()  # Turn on interactive mode for real-time updates
fig, ax = plt.subplots()

print("Camera is opening. Please wait...")
time.sleep(2)  # Wait for 2 seconds to ensure the camera initializes

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Predict age and gender using DeepFace models
        age, gender = predict_age_gender(face)

        # Draw bounding box and label for face detection
        if age and gender:
            label = f'Age: {int(age)}, Gender: {gender}'
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Perform YOLOv8 object detection
    results = yolo_model(frame)

    # Process the YOLOv8 results
    for result in results:
        # YOLOv8 results are typically returned as a list of dictionaries
        boxes = result.boxes.xywh.numpy()  # Get bounding boxes in (x, y, width, height) format
        scores = result.boxes.conf.numpy()  # Get confidence scores
        labels = result.boxes.cls.numpy()  # Get class labels

        # Loop through detected objects
        for box, score, label in zip(boxes, scores, labels):
            x, y, w, h = box
            label_name = yolo_model.names[int(label)]  # Get the class name from label index
            cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} ({score:.2f})", (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Convert the frame from BGR to RGB for displaying with matplotlib
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    ax.imshow(rgb_frame)
    ax.axis('off')
    plt.draw()
    plt.pause(0.001)  # Add a short pause to ensure the image updates in real-time

    # Exit on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release capture
cap.release()
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot open
