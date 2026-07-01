import tkinter as tk
import threading
import cv2
import math
import numpy as np
import platform
import os
import time
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
import pyttsx3

# ------------------- Text-to-Speech -------------------
engine = pyttsx3.init()

def speak(message):
    engine.say(message)
    engine.runAndWait()

# ------------------- Beep -------------------
def beep():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(1000, 200)
    else:
        os.system('echo -e "\a"')

# ------------------- YOLO Model -------------------
model = YOLO("bestFirst.pt")  

# ------------------- Safe Zone -------------------
safe_zone = []
PIXELS_PER_METER = 100

# ------------------- PPE Classes -------------------
# ⚠ CHANGE THESE ACCORDING TO YOUR YOLO TRAINING
PPE_CLASSES = ["helmet", "mask", "vest", "gloves"]

# ------------------- Face Recognition Database -------------------
db_path = "C:\\Users\\DELL\\Desktop\\DatasetfaaceRecognition"
print("Building face embeddings database...")

embeddings_db = {}

for person_folder in os.listdir(db_path):
    person_path = os.path.join(db_path, person_folder)
    if os.path.isdir(person_path):
        embeddings_db[person_folder] = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            emb = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embeddings_db[person_folder].append(emb)

print("✅ Face embeddings database ready!")

# ------------------- Alert Timing -------------------
alert_timers = {}
ALERT_INTERVAL = 0.2  # seconds
face_cache = {}  # Cache recognized faces
RECOGNITION_FRAME_SKIP = 5  # Do face recognition every N frames
frame_count = 0

def recognize_face(face_img, embeddings_db, threshold=0.4):
    try:
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        emb = DeepFace.represent(rgb_face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    except:
        return "Unknown"
    
    min_dist = float("inf")
    identity = "Unknown"

    for name, db_embs in embeddings_db.items():
        for db_emb in db_embs:
            dist = cosine(emb, db_emb)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                identity = name

    return identity

# ------------------- Process Frame -------------------
def process_frame(frame):
    global safe_zone, alert_timers, face_cache, frame_count
    
    frame_count += 1
    results = model(frame)[0]
    person_centers = []
    total_persons = 0
    persons_outside_zone = 0
    current_time = time.time()

    # ---- Draw Safe Zone ----
    if safe_zone:
        cv2.polylines(frame, [np.array(safe_zone, np.int32)], True, (0,255,255), 2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

        if label == "person":
            total_persons += 1
            person_centers.append((cx, cy))

            # Check safe zone violation
            if safe_zone:
                inside = cv2.pointPolygonTest(np.array(safe_zone, np.int32), (cx, cy), False)
                if inside < 0:
                    persons_outside_zone += 1
                    beep()

            # Extract person ROI
            person_roi = frame[y1:y2, x1:x2]
            roi_hash = hash(person_roi.tobytes())

            # ------- FACE RECOGNITION (with caching & frame skipping) -------
            if frame_count % RECOGNITION_FRAME_SKIP == 0:
                name = recognize_face(person_roi, embeddings_db)
                face_cache[roi_hash] = name
            else:
                name = face_cache.get(roi_hash, "Unknown")

            # ------- PPE DETECTION (optimized) -------
            wearing_ppe = False

            for b in results.boxes:
                cid = int(b.cls[0])
                lbl = model.names[cid].lower()

                if lbl in PPE_CLASSES:
                    px1, py1, px2, py2 = map(int, b.xyxy[0])

                    # Check overlap with person box (optimized)
                    if px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1:
                        wearing_ppe = True
                        break

            # Determine box color based on PPE safety
            if wearing_ppe:
                color = (0, 255, 0)  # Green - SAFE (wearing PPE)
            else:
                color = (0, 0, 255)  # Red - UNSAFE (not wearing PPE)

            # Draw box with safety color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ------- ALERT LOGIC -------
            if name != "Unknown" and not wearing_ppe:
                last = alert_timers.get(name, 0)
                if current_time - last > ALERT_INTERVAL:
                    threading.Thread(
                        target=speak,
                        args=(f"{name}, wear your personal protective equipment",),
                        daemon=True
                    ).start()
                    alert_timers[name] = current_time
        else:
            # Draw non-person boxes (PPE items) with safety color
            # Green = detected PPE (safe), Red = missing critical PPE (unsafe)
            color = (0, 255, 0)  # Green for detected PPE items (safe)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Distance display between people
    for i in range(len(person_centers)):
        for j in range(i+1, len(person_centers)):
            x1, y1 = person_centers[i]
            x2, y2 = person_centers[j]
            dist = math.hypot(x2 - x1, y2 - y1)
            meters = dist / PIXELS_PER_METER
            cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            mid = ((x1 + x2)//2, (y1 + y2)//2 - 10)
            cv2.putText(frame, f"{meters:.2f} m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.putText(frame, f"Total Persons: {total_persons}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Outside Zone: {persons_outside_zone}", (50, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    return frame

# ------------------- Draw Safe Zone -------------------
drawing = False
def set_safe_zone(event, x, y, flags, param):
    global drawing, safe_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        safe_zone.append((x, y))
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        safe_zone[-1] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def define_safe_zone():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Draw Safe Zone - Click points, press Enter to finish")
    cv2.setMouseCallback("Draw Safe Zone - Click points, press Enter to finish", set_safe_zone)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if safe_zone:
            cv2.polylines(frame, [np.array(safe_zone, np.int32)], False, (0,255,255), 2)
            for pt in safe_zone:
                cv2.circle(frame, pt, 5, (0,255,255), -1)

        cv2.imshow("Draw Safe Zone - Click points, press Enter to finish", frame)
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- Webcam Loop -------------------
def run_webcam():
    global frame_count
    alert_timers.clear()
    face_cache.clear()
    frame_count = 0
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow("Webcam Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def start_webcam():
    define_safe_zone()
    threading.Thread(target=run_webcam, daemon=True).start()

# ------------------- Tkinter GUI -------------------
root = tk.Tk()
root.title("YOLO + Face Recognition PPE Alert")
root.geometry("400x200")

lbl = tk.Label(root, text="YOLO + DeepFace PPE Detection", font=("Arial", 14))
lbl.pack(pady=20)

btn1 = tk.Button(root, text="Start Live Webcam", command=start_webcam, width=25, height=2)
btn1.pack(pady=10)

root.mainloop()
