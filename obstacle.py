import cv2
import threading
import pyttsx3
import time
from ultralytics import YOLO

# --- Setup ---
model = YOLO("yolov8n.pt")  
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # speaking speed - lower is slower and clearer

# Objects that actually matter for pedestrian navigation
RELEVANT_OBJECTS = {
    "person", "car", "motorcycle", "bicycle",
    "bus", "truck", "chair", "dining table", "stop sign", "pole",
    "door", "trash bin"
}

CONFIDENCE_THRESHOLD = 0.5
ALERT_COOLDOWN = 3  # seconds between alerts so it doesn't spam
last_alert_time = 0


def estimate_proximity(box_height, frame_height):
    """Rough distance estimate based on bounding box size."""
    ratio = box_height / frame_height
    if ratio > 0.6:
        return "very close"
    elif ratio > 0.35:
        return "nearby"
    else:
        return "ahead"


def estimate_direction(box_center_x, frame_width):
    """Tell user which way to move based on where obstacle is in frame."""
    left_zone = frame_width * 0.35
    right_zone = frame_width * 0.65
    if box_center_x < left_zone:
        return "on your left"
    elif box_center_x > right_zone:
        return "on your right"
    else:
        return "straight ahead — move aside"


def speak(text):
    print(f"[ALERT] {text}")
    def _speak():
        e = pyttsx3.init()
        e.setProperty("rate", 150)
        e.say(text)
        e.runAndWait()
    t = threading.Thread(target=_speak)
    t.start()


def run():
    global last_alert_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Eyeris obstacle detection running. Press Q to quit.")
    speak("Eyeris is active. Scanning for obstacles.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        results = model(frame, verbose=False)

        current_time = time.time()
        closest_alert = None
        highest_ratio = 0

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                label = model.names[int(box.cls)]

                if confidence < CONFIDENCE_THRESHOLD or label not in RELEVANT_OBJECTS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_height = y2 - y1
                box_center_x = (x1 + x2) / 2
                ratio = box_height / frame_height

                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # Track the closest/most prominent object for alert
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    proximity = estimate_proximity(box_height, frame_height)
                    direction = estimate_direction(box_center_x, frame_width)
                    closest_alert = f"{label} {proximity}, {direction}"

        # Speak alert only if cooldown has passed
        if closest_alert and (current_time - last_alert_time) > ALERT_COOLDOWN:
            speak(closest_alert)
            last_alert_time = current_time

        cv2.imshow("Eyeris - Obstacle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Eyeris stopped.")


if __name__ == "__main__":
    run()