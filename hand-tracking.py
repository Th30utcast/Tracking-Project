import mediapipe as mp
import cv2
import pyautogui
import math
import numpy as np
import threading
import tkinter as tk

# Initialize Mediapipe and pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
pyautogui.FAILSAFE = False

# Function to determine if a finger is up or down
def is_finger_up(landmarks, finger_tip_id, finger_pip_id):
    # Finger is up if the tip y-coordinate is less than PIP y-coordinate
    return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y

# Function to determine if the thumb is up
def is_thumb_up(landmarks, hand_label):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    if hand_label == 'Left':
        return thumb_tip.x > thumb_ip.x
    else:
        return thumb_tip.x < thumb_ip.x

# Function to show the fullscreen black window with text
def show_black_screen():
    global root
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.configure(bg='black')
    label = tk.Label(root, text="No Face Detected", fg="white", bg="black", font=("Arial", 48))
    label.pack(expand=True)
    root.mainloop()

# Initialize variables for the black screen
black_screen_thread = None
black_screen_visible = False

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    max_num_hands=2) as hands, \
     mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:

    # State variables for each hand
    pinching_left = False
    holding_left = False
    prev_right_hand_position = None
    smoothed_right_hand_x = None
    smoothed_right_hand_y = None
    alpha = 0.2  # Smoothing factor for exponential moving average
    sensitivity = 5.0  # Sensitivity adjustment for cursor movement
    screen_width, screen_height = pyautogui.size()

    while True:
        data, image = cap.read()
        if not data:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face detection
        face_results = face_detection.process(image_rgb)

        if not face_results.detections:
            # No face detected, show black screen
            if not black_screen_visible:
                black_screen_visible = True
                black_screen_thread = threading.Thread(target=show_black_screen)
                black_screen_thread.start()
        else:
            # Face detected, hide black screen if it's visible
            if black_screen_visible:
                black_screen_visible = False
                # Close the Tkinter window
                root.quit()
                black_screen_thread.join()

        if black_screen_visible:
            # Skip processing if black screen is visible
            continue

        results = hands.process(image_rgb)

        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_index, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label

                landmarks = hand_landmarks.landmark

                # Finger tip and PIP indices
                finger_tips = [mp_hands.HandLandmark.THUMB_TIP,
                               mp_hands.HandLandmark.INDEX_FINGER_TIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                               mp_hands.HandLandmark.RING_FINGER_TIP,
                               mp_hands.HandLandmark.PINKY_TIP]

                finger_pips = [mp_hands.HandLandmark.THUMB_IP,
                               mp_hands.HandLandmark.INDEX_FINGER_PIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                               mp_hands.HandLandmark.RING_FINGER_PIP,
                               mp_hands.HandLandmark.PINKY_PIP]

                # Determine finger states
                fingers_up = []
                for tip_id, pip_id in zip(finger_tips, finger_pips):
                    is_up = is_finger_up(landmarks, tip_id, pip_id)
                    fingers_up.append(is_up)

                # Update thumb state using the new function
                thumb_is_up = is_thumb_up(landmarks, hand_label)
                fingers_up[0] = thumb_is_up  # Update thumb state

                # For Left Hand - Simulate Clicks
                if hand_label == 'Left':
                    # Check for pinch gesture: Thumb and index finger are up, others down
                    is_pinch_gesture = fingers_up[0] and fingers_up[1] and not any(fingers_up[2:])
                    if is_pinch_gesture:
                        # Calculate the distance between thumb tip and index finger tip
                        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                        index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        distance = math.hypot(thumb_tip.x - index_finger_tip.x, thumb_tip.y - index_finger_tip.y)

                        pinch_threshold = 0.05  # Adjust based on your setup

                        if distance < pinch_threshold:
                            if not pinching_left:
                                pyautogui.mouseDown()
                                pinching_left = True
                                holding_left = True
                        else:
                            if holding_left:
                                pyautogui.mouseUp()
                                holding_left = False
                            elif pinching_left:
                                pyautogui.click()
                            pinching_left = False
                        # Visual feedback for left hand pinch gesture
                        cv2.putText(image, 'Left Hand: Clicking', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)
                    else:
                        # Fingers are not in pinch gesture
                        if holding_left:
                            pyautogui.mouseUp()
                            holding_left = False
                        pinching_left = False

                # For Right Hand - Control Cursor Movement with Closed Fist
                if hand_label == 'Right':
                    # Check if all fingers are down (closed fist)
                    is_fist = not any(fingers_up)

                    if is_fist:
                        # Get current hand position using wrist landmark
                        hand_x = landmarks[mp_hands.HandLandmark.WRIST].x
                        hand_y = landmarks[mp_hands.HandLandmark.WRIST].y

                        # Apply exponential moving average for smoothing
                        if smoothed_right_hand_x is None or smoothed_right_hand_y is None:
                            smoothed_right_hand_x = hand_x
                            smoothed_right_hand_y = hand_y
                        else:
                            smoothed_right_hand_x = alpha * hand_x + (1 - alpha) * smoothed_right_hand_x
                            smoothed_right_hand_y = alpha * hand_y + (1 - alpha) * smoothed_right_hand_y

                        if prev_right_hand_position is not None:
                            # Calculate movement
                            dx = (smoothed_right_hand_x - prev_right_hand_position[0]) * screen_width * sensitivity
                            dy = (smoothed_right_hand_y - prev_right_hand_position[1]) * screen_height * sensitivity

                            # Update mouse position
                            current_mouse_x, current_mouse_y = pyautogui.position()
                            new_mouse_x = current_mouse_x + dx
                            new_mouse_y = current_mouse_y + dy

                            # Clamp to screen size
                            new_mouse_x = max(0, min(screen_width - 1, new_mouse_x))
                            new_mouse_y = max(0, min(screen_height - 1, new_mouse_y))

                            pyautogui.moveTo(new_mouse_x, new_mouse_y)

                        prev_right_hand_position = (smoothed_right_hand_x, smoothed_right_hand_y)

                        # Visual feedback for right hand cursor control
                        cv2.putText(image, 'Right Hand: Cursor Control Active', (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2)
                    else:
                        prev_right_hand_position = None
                        smoothed_right_hand_x = None
                        smoothed_right_hand_y = None

                # Draw the hand annotations on the image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            # No hands detected
            if holding_left:
                pyautogui.mouseUp()
                holding_left = False
            pinching_left = False
            prev_right_hand_position = None
            smoothed_right_hand_x = None
            smoothed_right_hand_y = None

        # Display the image
        cv2.imshow('Hand Tracking', image)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if black_screen_visible:
                # Close the Tkinter window if it's open
                root.quit()
                black_screen_thread.join()
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
