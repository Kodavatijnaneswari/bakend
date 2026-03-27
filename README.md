# Bone Abnormality Detection System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Django](https://img.shields.io/badge/django-%23092e20.svg?style=for-the-badge&logo=django&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v8-blueviolet?style=for-the-badge)

A professional medical imaging application designed for the automated detection of bone abnormalities (fractures) in X-ray images. The system features a modern Django-based web dashboard and a React Native mobile application, powered by a state-of-the-art YOLOv8 object detection model.

## 🌟 Features

- **High-Accuracy Detection**: Utilizes YOLOv8 for precise identification and localization of bone fractures.
- **Modern Web Dashboard**: Professional "Medical Glassmorphism" UI for easy case management and analysis.
- **Mobile Integration**: Dedicated React Native app for on-the-go diagnostic reviews.
- **Diagnostic History**: Persistent storage of detection results with clinical reporting features.
- **Real-time Feedback**: Interactive previews and visual feedback during model inference.
- **Export Capabilities**: Generate PDF reports for clinical documentation.

## 🛠️ Tech Stack

- **Backend**: Django, Django REST Framework
- **Frontend**: Vanilla CSS (Modern UI Design), React Native (Mobile)
- **AI/ML**: YOLOv8 (Ultralytics), PyTorch, OpenCV
- **Database**: SQLite (Default) / PostgreSQL (Recommended for production)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js & npm (for mobile app)
- Virtual Environment (recommended)

### Backend Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Run the server**:
   ```bash
   python manage.py runserver
   ```

### Mobile App Setup
1. Navigate to the mobile app directory:
   ```bash
   cd BoneMobileApp
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the app:
   ```bash
   npx react-native run-android # or run-ios
   ```

## 📂 Project Structure

```text
├── Bone_Abnormality_Detection/ # Project settings
├── admins/                     # Admin dashboard app
├── users/                      # User management app
├── BoneMobileApp/              # React Native mobile application
├── media/                      # Uploaded images and detection results
├── scripts/                    # Utility scripts for training and testing
│   ├── check_classes.py
│   ├── test_inference.py
│   ├── train_robust_model.py
│   └── train_v2.py
├── static/                     # CSS, JS, and Images
├── templates/                  # HTML templates
├── manage.py                   # Django management script
└── requirements.txt            # Project dependencies
```

## 📖 Usage

1. Log in to the Web Dashboard.
2. Upload an X-ray image via the "New Detection" section.
3. Wait for the AI model to process the image.
4. Review the detected abnormalities and download the diagnostic report.

## 📄 License
Individual property of the developer. Use for academic or clinical research purposes only.
"# bone" 
"# bone" 
"# bone" 
