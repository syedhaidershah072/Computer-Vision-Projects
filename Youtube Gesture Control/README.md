# ✋🎥 YouTube Hand Gesture Control with Python

Control YouTube playback **without touching your keyboard** — just use your hand gestures!  
This project uses **MediaPipe**, **OpenCV**, and **PyAutoGUI** to recognize hand gestures from your webcam and map them to YouTube player controls like **play**, **pause**, **seek**, and **volume**.

---

## 📌 Features

- ✅ Real-time hand tracking using MediaPipe
- ✅ Play / Pause video with open/closed fist
- ✅ Seek forward and backward using thumb + index / thumb + pinky gestures
- ✅ Control volume with gestures (rock for down, 2-finger V for up)
- ✅ On-screen feedback showing detected gesture label
- ✅ Touchless control of YouTube videos (and other media apps)

---

## 🛠 Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- PyAutoGUI

Install required libraries with:

```bash
pip install opencv-python mediapipe pyautogui
