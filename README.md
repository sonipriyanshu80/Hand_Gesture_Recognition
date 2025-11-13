# Hand Gesture Recognition using OpenCV and MediaPipe

A simple Python program that recognizes hand gestures in real-time using your webcam.

## Features

- Real-time hand gesture detection using webcam
- Detects 21 hand landmarks using MediaPipe
- Recognizes common gestures:
  - Fist (0 fingers)
  - One Finger
  - Victory (2 fingers)
  - Open Palm (5 fingers)
  - Thumbs Up

## Requirements

- Python 3.7 to 3.12 (MediaPipe doesn't support Python 3.13 yet)
- Webcam
- Internet connection (for initial package installation)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```
or
```bash
python -m pip install -r requirements.txt
```

## Usage

Run the program:
```bash
python main.py
```

- The webcam window will open
- Show your hand in front of the camera
- Hand landmarks will be drawn on the screen
- Gesture name and finger count will be displayed
- Press `q` key to quit

## Project Structure

```
hand-gesture-recognition/
├── README.md
├── requirements.txt
├── main.py
├── .gitignore
├── .github/
│   └── workflows/
│       └── static.yml
└── docs/
    └── project_description.txt
```

## How It Works

The program uses MediaPipe to detect hand landmarks. It counts fingers by comparing the positions of finger tips with their PIP (Proximal Interphalangeal) joints. Based on the finger count, it identifies different gestures.

## Notes

- Make sure you have good lighting for better detection
- Keep your hand clearly visible in the frame
- The program detects only one hand at a time

## Author

Student Project - 5th Semester

