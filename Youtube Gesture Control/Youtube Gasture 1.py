import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

# Start webcam
cap = cv2.VideoCapture(0)

# Variables
prev_gesture = None
last_action_time = 0
cooldown = 1.5
current_display_gesture = ""

# Finger landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

# ----------------------------- #
# Helper Functions
# ----------------------------- #

def get_finger_states(landmarks):
    return [1 if landmarks[tip].y < landmarks[pip].y else 0
            for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]

def gesture_name(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Pause"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Play/Resume"
    elif fingers == [1, 1, 0, 0, 0]:
        return "Seek Forward"
    elif fingers == [1, 0, 0, 0, 1]:
        return "Seek Backward"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Volume Up"
    elif fingers == [1, 1, 0, 0, 1]:
        return "Volume Down"
    return ""

def perform_action(gesture):
    global prev_gesture, last_action_time
    now = time.time()
    if gesture == prev_gesture and now - last_action_time < cooldown:
        return

    if gesture == "Pause":
        pyautogui.press('k')
        print("Pause")
    elif gesture == "Play/Resume":
        pyautogui.press('k')
        print("Play/Resume")
    elif gesture == "Seek Forward":
        pyautogui.press('l')
        print("Seek Forward")
    elif gesture == "Seek Backward":
        pyautogui.press('j')
        print("Seek Backward")
    elif gesture == "Volume Up":
        pyautogui.press('volumeup')
        print("Volume Up")
    elif gesture == "Volume Down":
        pyautogui.press('volumedown')
        print("Volume Down")

    prev_gesture = gesture
    last_action_time = now

# ----------------------------- #
# Main Loop
# ----------------------------- #

print("✅ Gesture Control Started — Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_display_gesture = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            fingers = get_finger_states(landmarks)
            gesture = gesture_name(fingers)

            if gesture:
                current_display_gesture = gesture
                perform_action(gesture)

    # Display gesture name on screen
    if current_display_gesture:
        cv2.putText(frame, f"Gesture: {current_display_gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YouTube Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
