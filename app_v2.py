import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np

# Load YOLOv8 model for face detection
model = YOLO('yolov8n.pt')  

# Function to preprocess image for low-light conditions
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    hist_eq = cv2.equalizeHist(gray)
    # Convert back to BGR
    bgr_eq = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
    return bgr_eq

# Function to predict age, gender, and expression
def predict_age_gender_expression(image):
    try:
        # Convert image to RGB format as DeepFace expects this format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Analyze the face with DeepFace
        analysis = DeepFace.analyze(rgb_image, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        # If analysis is a list, get the first element
        if isinstance(analysis, list):
            analysis = analysis[0]
        age = analysis.get('age', "Unknown")
        gender = analysis.get('gender', "Unknown")
        expression = max(analysis.get('emotion', {}), key=analysis.get('emotion', {}).get, default="Unknown")
    except Exception as e:
        print(f"Error in DeepFace analysis: {e}")
        age = "Unknown"
        gender = "Unknown"
        expression = "Unknown"
    return age, gender, expression

# Open webcam
cap = cv2.VideoCapture(0)

# Create a named window and set it to fullscreen
cv2.namedWindow('Age, Gender, and Expression Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Age, Gender, and Expression Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for better visibility in low light
    frame = preprocess_image(frame)

    # Perform face detection using YOLOv8
    try:
        # Run inference
        results = model(frame)
        # Extract bounding boxes from the results
        boxes = results[0].boxes
        # Convert detection results to a list of detections
        detections = boxes.xyxy.numpy()  # Convert to numpy array for easier manipulation
        confidences = boxes.conf.numpy()
        class_ids = boxes.cls.numpy()

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections[i])
            score = confidences[i]
            class_id = int(class_ids[i])

            # Assuming 'class_id == 0' corresponds to faces in your dataset
            if class_id == 0:  # Adjust based on your dataset
                face = frame[y1:y2, x1:x2]

                # Predict age, gender, and expression for the detected face
                age, gender, expression = predict_age_gender_expression(face)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Prepare text to display
                text_lines = [
                    f'Age: {age}',
                    f'Gender: {gender}',
                    f'Expression: {expression}'
                ]

                # Define font scale and thickness
                font_scale = 0.5
                thickness = 1

                # Display each line of text
                y_offset = y1 - 10
                for line in text_lines:
                    cv2.putText(frame, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), thickness)
                    y_offset += 20  # Adjust spacing between lines

    except Exception as e:
        print(f"Error during YOLOv8 inference: {e}")

    # Display the frame in full screen
    cv2.imshow('Age, Gender, and Expression Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
