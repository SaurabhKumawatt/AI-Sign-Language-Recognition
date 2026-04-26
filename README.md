# 🤟 AI-Based Sign Language Recognition — Web Application

> *Bridging communication gaps through real-time gesture intelligence.*

---

## 📌 Overview

A real-time AI-powered web application that recognizes sign language gestures via webcam and converts them into **text + speech** — making communication more accessible for hearing and speech-impaired individuals.

Built with computer vision (MediaPipe) and machine learning (Random Forest), it runs entirely in the browser through a lightweight Flask backend.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🎥 Live Webcam Input | Real-time video capture directly in browser |
| ✋ Hand Detection | 21-point landmark detection via MediaPipe |
| 🧠 ML Gesture Recognition | Trained Random Forest model for fast predictions |
| 🔤 Text Output | Gesture name displayed on-screen instantly |
| 📊 Confidence Score | Live probability score for each prediction |
| 🔊 Voice Output | Text-to-Speech using gTTS + Pygame |
| 🌐 Web Interface | Flask-powered, no installation needed for users |
| 📱 Responsive UI | Mobile-friendly design |

---

## 🛠️ Tech Stack

**Backend**
- Python, Flask
- OpenCV, MediaPipe
- NumPy, Scikit-learn
- gTTS, Pygame

**Frontend**
- HTML5, CSS3, JavaScript

---

## ⚙️ How It Works

```
Webcam Feed
    ↓
MediaPipe → 21 Hand Landmarks Detected
    ↓
Landmark Coordinates → Feature Vector
    ↓
Normalization
    ↓
Random Forest Model → Gesture Predicted
    ↓
UI Display (Text + Confidence Score) + TTS (Voice Output)
```

---

## 🤖 ML Model Details

- **Algorithms Evaluated:** K-Nearest Neighbors (KNN), Random Forest
- **Final Model:** Random Forest *(selected for superior accuracy)*
- **Dataset:** ~3,400+ samples across 10 gesture classes

### 📊 Accuracy Results

| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors | ~99.85% |
| **Random Forest** | **~100%** ✅ |

> ⚠️ **Note:** Real-world accuracy may vary depending on lighting conditions, hand angle, distance, and background clutter.

---

## 📁 Project Structure

```
AI-Sign-Language-Recognition/
│
├── app.py                    # Flask backend (main entry point)
├── predict.py                # Local real-time prediction script
├── train_model.py            # Model training pipeline
├── collect_data.py           # Data collection utility
│
├── model/
│   └── gesture_model.pkl     # Trained Random Forest model
│
├── dataset/                  # Gesture training data
│
├── webapp/
│   └── templates/
│       └── index.html        # Frontend UI
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/SaurabhKumawatt/AI-Sign-Language-Recognition.git
cd AI-Sign-Language-Recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

### 4. Open in Browser

```
http://127.0.0.1:5000
```

---

## 🎮 Usage

1. Allow webcam access when prompted
2. Show a hand gesture in front of the camera
3. The system will automatically:
   - Detect and highlight hand landmarks
   - Display the predicted gesture name
   - Show a real-time confidence score
   - Speak the gesture aloud via voice output

---

## ⚠️ Known Limitations

- Supports **10–20 static gestures** only (no dynamic/motion gestures)
- Best performance in **well-lit environments**
- Does **not** support full sentence or phrase recognition
- Currently optimized for **single-hand** gestures only
- No real-time multi-hand tracking

---

## 🔮 Future Roadmap

- [ ] Full sentence and phrase recognition
- [ ] Deep learning upgrade (CNN + LSTM for sequential gestures)
- [ ] Mobile app (Android/iOS)
- [ ] Multi-language sign language support
- [ ] Sign-to-animated avatar system
- [ ] Dataset expansion with community contributions

---

## 👨‍💻 Author

**Saurrabh Kumawat**  
BTech CSE | CTO & Co-Founder @ StraviX  
[GitHub](https://github.com/SaurabhKumawatt)

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) by Google — for hand landmark detection
- [OpenCV](https://opencv.org/) — for real-time video processing
- [Scikit-learn](https://scikit-learn.org/) — for ML model training & evaluation

---

<div align="center">

**⭐ If this project helped you, consider giving it a star!**

</div>
