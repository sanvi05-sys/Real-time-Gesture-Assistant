import cv2
import os

# Ask user for gesture name
gesture = input("Enter gesture name (e.g., thumbs_up): ").strip()
output_dir = f"data/raw_gestures/{gesture}"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print(f"Press 's' to save images of {gesture} (50 needed)...")

while True:
    ret, frame = cap.read()
    cv2.putText(frame, f"Gesture: {gesture} ({count}/50)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Press 's' to save, 'q' to quit", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"{output_dir}/{count:03d}.jpg", frame)
        print(f"Saved {count+1}/50")
        count += 1
    elif key == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Collected {count} images for {gesture} in {output_dir}")