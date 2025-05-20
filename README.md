# Fruit Ninja Game

A simple real-time Fruit Ninja–style game using hand tracking and object detection.

---

## Description

**FruitNinjaGame** is a Python application that uses:

- **MediaPipe** for real-time hand pose estimation  
- **YOLO** (via the `ultralytics` package) for object-detection overlays  
- **OpenCV** for video capture and rendering  

Fruits are generated at random positions, speeds, and colors. You “slash” them by moving your index finger through them in front of your camera. Slice enough fruits to win, but don’t miss too many, or it’s Game Over!

> ⚠️ _This project was created for personal entertainment and is not maintained. It does not include elaborate comments or docstrings._

---

## Features

- **Real-time hand tracking** with MediaPipe  
- **YOLO bounding boxes** for extra visual flair  
- Random fruit generation (position, speed, color)  
- Score & miss tracking, with win/lose conditions  
- Configurable camera index, thresholds, and game parameters  

---

## Requirements

- **Python 3.7+**  
- **OpenCV** (`opencv-python`)  
- **MediaPipe** (`mediapipe`)  
- **Ultralytics YOLO** (`ultralytics`)  
- **YOLO model weights** file: `yolo11l.pt`  

---

## Installation

1. **Clone the repo (or copy the files):**
   ```bash
   git clone https://github.com/yourusername/fruit-ninja-game.git
   cd fruit-ninja-game

