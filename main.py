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
model = YOLO("bestFirst.pt")  # replace with your trained weights

# ------------------- Safe Zone -------------------
safe_zone = [(200, 150), (1100, 150), (1100, 550), (200, 550)]
PIXELS_PER_METER = 100

# ------------------- Face Recognition -------------------
db_path = "C:\\Users\\DELL\\Desktop\\DatasetfaaceRecognition"
print("Building face embeddings database...")
embeddings_db = {}  # name -> list of embeddings
for person_folder in os.listdir(db_path):
    person_path = os.path.join(db_path, person_folder)
    if os.path.isdir(person_path):
        embeddings_db[person_folder] = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            emb = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embeddings_db[person_folder].append(emb)
print("✅ Face embeddings database ready!")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------- Alert Timing -------------------
alert_timers = {}  # name -> last alert time
ALERT_INTERVAL = 10  # seconds between repeated voice alerts

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
    global safe_zone, alert_timers
    results = model(frame)[0]
    person_centers = []
    total_persons, persons_outside_zone = 0, 0
    current_time = time.time()

    # Draw Safe Zone
    if safe_zone:
        cv2.polylines(frame, [np.array(safe_zone, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(frame, "Safe Zone", (safe_zone[0][0], safe_zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

        color = (0,255,0) if label.lower()=="person" else (128,128,128)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (cx,cy), 5, (0,255,255), -1)

        if label.lower()=="person":
            total_persons +=1
            person_centers.append((cx,cy))

            if safe_zone:
                inside = cv2.pointPolygonTest(np.array(safe_zone,np.int32),(cx,cy),False)
                if inside < 0:
                    persons_outside_zone +=1
                    beep()

            # ---------------- Face Recognition ----------------
            face_img = frame[y1:y2, x1:x2]
            name = recognize_face(face_img, embeddings_db)
            cv2.putText(frame, name, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,0) if name!="Unknown" else (0,0,255), 2)

            # Voice alert at interval
            if name != "Unknown":
                last_alert = alert_timers.get(name, 0)
                if current_time - last_alert > ALERT_INTERVAL:
                    threading.Thread(target=speak, args=(f"{name}, wear your personal protective equipment",), daemon=True).start()
                    alert_timers[name] = current_time

    # Distance lines
    for i in range(len(person_centers)):
        for j in range(i+1, len(person_centers)):
            x1, y1 = person_centers[i]
            x2, y2 = person_centers[j]
            pixel_distance = math.hypot(x2-x1, y2-y1)
            distance_m = pixel_distance/PIXELS_PER_METER
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
            mid = ((x1+x2)//2,(y1+y2)//2 -10)
            cv2.putText(frame, f"{distance_m:.2f} m", mid, cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.putText(frame, f"Total Persons: {total_persons}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(frame, f"Outside Zone: {persons_outside_zone}", (50,85), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    return frame

# ------------------- Run Webcam -------------------
def run_webcam():
    global alert_timers
    alert_timers.clear()
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

# ------------------- Tkinter GUI -------------------
def start_webcam():
    threading.Thread(target=run_webcam, daemon=True).start()

root = tk.Tk()
root.title("YOLO + Face Recognition PPE Alert")
root.geometry("400x200")

lbl = tk.Label(root, text="YOLO + DeepFace PPE Detection", font=("Arial", 14))
lbl.pack(pady=20)

btn1 = tk.Button(root, text="Start Live Webcam", command=start_webcam, width=25, height=2)
btn1.pack(pady=10)

root.mainloop()
