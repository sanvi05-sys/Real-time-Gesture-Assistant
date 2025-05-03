import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyautogui
import os
import time
import pygame
from datetime import datetime

# Initialize pygame for audio feedback
pygame.mixer.init()
pygame.mixer.music.load("notification.mp3")  # Add a short audio file

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

# Load trained model and gesture names
model = load_model('models/gesture_model.h5')
gesture_names = np.load('data/processed/gesture_names.npy')
print("Loaded model for gestures:", list(gesture_names))

# Define actions for each gesture
ACTIONS = {
    'thumbs_up': {
        'name': "Play Music",
        'function': lambda: pygame.mixer.music.play() if not pygame.mixer.music.get_busy() else None,
        'color': (0, 255, 0)  # Green
    },
    'thumbs_down': {
        'name': "Stop Music",
        'function': lambda: pygame.mixer.music.stop(),
        'color': (0, 0, 255)  # Red
    },
    'fist': {
        'name': "Take Screenshot",
        'function': lambda: pyautogui.screenshot(f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
        'color': (255, 0, 0)  # Blue
    },
    'L_sign': {  # Changed from 'rock' to 'L_sign'
        'name': "Open Calculator",
        'function': lambda: os.system('calc.exe'),
        'color': (255, 255, 0)  # Yellow
    },
    'open_hand': {
        'name': "Show Time",
        'function': lambda: print("Current time:", datetime.now().strftime("%H:%M:%S")),
        'color': (0, 255, 255)  # Cyan
    },
    'v_sign': {
        'name': "Open Browser",
        'function': lambda: os.system('start chrome https://google.com'),
        'color': (255, 0, 255)  # Magenta
    }
}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Variables for action cooldown and status
last_action_time = 0
ACTION_COOLDOWN = 2  # seconds
current_status = "Show gesture to camera..."
music_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame")
        break
    
    # Mirror the frame and convert color space
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract and normalize landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks).reshape(1, -1)
            
            # Predict gesture
            prediction = model.predict(landmarks, verbose=0)
            gesture_id = np.argmax(prediction)
            confidence = np.max(prediction)
            gesture = gesture_names[gesture_id]
            
            # Only consider high-confidence predictions
            if confidence > 0.9:
                action = ACTIONS.get(gesture)
                if action:
                    # Display gesture info
                    cv2.putText(frame, f"{gesture} ({confidence:.2f})", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, action['color'], 2)
                    
                    # Execute action with cooldown
                    current_time = time.time()
                    if current_time - last_action_time > ACTION_COOLDOWN:
                        print(f"Action: {action['name']}")
                        action['function']()
                        last_action_time = current_time
                        current_status = f"Executed: {action['name']}"

    # Display system status
    cv2.putText(frame, current_status, (50, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('ISL Gesture Control', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Gesture recognition stopped")