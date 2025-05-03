import cv2
import numpy as np
import os
import mediapipe as mp
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Map gesture folders to class names
GESTURE_MAP = {
    'thumbs_up': 0,
    'thumbs_down': 1,
    'fist': 2,
    'L_sign': 3,
    'open_hand': 4,
    'v_sign': 5
}

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert to RGB and process
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    return None

def process_dataset():
    X, y = [], []
    
    for gesture_name, class_id in GESTURE_MAP.items():
        gesture_dir = os.path.join('data', 'raw_gestures', gesture_name)
        if not os.path.exists(gesture_dir):
            continue
            
        print(f"Processing {gesture_name}...")
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.png'))]
        
        for img_file in tqdm(image_files):
            landmarks = extract_landmarks(os.path.join(gesture_dir, img_file))
            if landmarks is not None:
                X.append(landmarks)
                y.append(class_id)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X.npy', np.array(X))
    np.save('data/processed/y.npy', np.array(y))
    np.save('data/processed/gesture_names.npy', np.array(list(GESTURE_MAP.keys())))
    
    print(f"Processed {len(X)} samples across {len(GESTURE_MAP)} gestures")

if __name__ == "__main__":
    process_dataset()