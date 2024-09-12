---

# Age, Gender, and Expression Detection with OpenCV

This repository contains a Python-based model that utilizes OpenCV and pre-trained deep learning models to detect a person's age, gender, and facial expressions. The model is designed for real-time analysis and can be integrated with live camera feeds or static images.

### Features:
- **Age Detection**: Predicts the age range of detected faces.
- **Gender Detection**: Classifies the gender of the detected individual as male or female.
- **Expression Detection**: Identifies the facial expression (e.g., happy, sad, angry) using the FER (Facial Expression Recognition) model.
- **Real-Time Video/Static Image Support**: Works with both live webcam feeds and static images.

### Dependencies:
- Python 3.x
- OpenCV
- TensorFlow
- Keras
- FER (Facial Expression Recognition)
  
### Installation:
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/age-gender-expression-detection.git
   ```
2. Install the required dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python tensorflow keras fer
   ```

3. Download the pre-trained models for age and gender detection:
   - **Age Model**: [age_net.caffemodel](https://github.com/GilLevi/AgeGenderDeepLearning)
   - **Gender Model**: [gender_net.caffemodel](https://github.com/GilLevi/AgeGenderDeepLearning)

4. Place the models in the `models/` directory.

### Usage:
To run the model, you can use the following script for real-time detection:
```bash
python detect_age_gender_expression.py
```

### Project Structure:
```bash
.
├── README.md
├── detect_age_gender_expression.py   # Main script for running the detection
├── models/                           # Directory for pre-trained models
│   ├── age_net.caffemodel
│   ├── deploy_age.prototxt
│   ├── gender_net.caffemodel
│   └── deploy_gender.prototxt
└── images/                           # Example images for testing
```

### Model Description:
- **Face Detection**: Uses OpenCV’s Haar cascades to detect faces in the image or video.
- **Age and Gender Detection**: Pre-trained Caffe models that estimate the age range and gender.
- **Expression Detection**: Uses the FER library to classify facial expressions such as happy, sad, angry, and more.

### Example Output:
- **Age**: 25-32
- **Gender**: Male
- **Expression**: Happy

### License:
This project is licensed under the MIT License.

---
