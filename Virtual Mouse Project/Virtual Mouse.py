import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

# Gesture state variables
click_start_time = None
click_times = []
click_cooldown = 0.5
scroll_mode = False
freeze_cursor = False

screen_w, screen_h = pyautogui.size()
print("\n✅ Smart Hand Mouse Control Started — Press ESC to exit.")

prev_screen_x, prev_screen_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger tips
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Determine fingers state
            fingers = [
                1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0
                for tip in [8, 12, 16, 20]
            ]

            # Distance between thumb and index
            dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

            # Freeze cursor when index & thumb close
            if dist < 0.04:
                if not freeze_cursor:
                    freeze_cursor = True
                    click_times.append(time.time())

                    # Double click check
                    if len(click_times) >= 2 and click_times[-1] - click_times[-2] < 0.4:
                        pyautogui.doubleClick()
                        cv2.putText(frame, "Double Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        click_times = []
                    else:
                        pyautogui.click()
                        cv2.putText(frame, "Single Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                freeze_cursor = False

            # Move mouse only when not frozen
            if not freeze_cursor:
                screen_x = int(index_tip.x * screen_w)
                screen_y = int(index_tip.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                prev_screen_x, prev_screen_y = screen_x, screen_y

            # Scroll mode detection: All fingers open
            if sum(fingers) == 4:
                scroll_mode = True
            else:
                scroll_mode = False

            # Scroll action (vertical movement of index)
            if scroll_mode:
                if index_tip.y < 0.4:
                    pyautogui.scroll(60)  # faster scroll
                    cv2.putText(frame, "Scroll Up", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif index_tip.y > 0.6:
                    pyautogui.scroll(-60)  # faster scroll
                    cv2.putText(frame, "Scroll Down", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Smart Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
