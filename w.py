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
from collections import deque

# ------------------- Text-to-Speech (Optimized) -------------------
engine = pyttsx3.init()
engine.setProperty('rate', 200)
speech_queue = deque(maxlen=5)
speech_lock = threading.Lock()

def speak(message):
    """Non-blocking speech with queue management"""
    def _speak():
        with speech_lock:
            if len(speech_queue) < 5:
                speech_queue.append(message)
                engine.say(message)
                engine.runAndWait()
                if speech_queue:
                    speech_queue.popleft()
    
    threading.Thread(target=_speak, daemon=True).start()

# ------------------- Beep -------------------
def beep():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(1000, 100)
    else:
        os.system('echo -e "\a"')

# ------------------- YOLO Model (Optimized) -------------------
print("Loading YOLO model...")
model = YOLO("bestFirst.pt")
model.fuse()
print("✅ YOLO model loaded!")

# Set inference parameters
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# ------------------- Safe Zone -------------------
safe_zone = []
PIXELS_PER_METER = 100

# ------------------- PPE Classes -------------------
PPE_CLASSES = ["helmet", "mask", "vest", "gloves"]

# Define colors for different detection types (BGR format)
DETECTION_COLORS = {
    "person": (0, 255, 0),      # Green
    "helmet": (0, 255, 255),    # Yellow
    "mask": (255, 0, 255),      # Magenta
    "vest": (0, 165, 255),      # Orange
    "gloves": (255, 255, 0),    # Cyan
    "default": (255, 128, 0)    # Blue (for other detections)
}

# ------------------- Face Recognition (Optimized) -------------------
db_path = "C:\\Users\\DELL\\Desktop\\DatasetfaaceRecognition"
print("Building face embeddings database...")
embeddings_db = {}
embeddings_array = {}

if os.path.exists(db_path):
    for person_folder in os.listdir(db_path):
        person_path = os.path.join(db_path, person_folder)
        if os.path.isdir(person_path):
            embeddings_db[person_folder] = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        emb = DeepFace.represent(
                            img_path=img_path,
                            model_name="Facenet",
                            enforce_detection=False,
                            detector_backend='opencv'
                        )[0]["embedding"]
                        embeddings_db[person_folder].append(emb)
                    except Exception as e:
                        print(f"⚠ Error processing {img_path}: {e}")
            
            if embeddings_db[person_folder]:
                embeddings_array[person_folder] = np.array(embeddings_db[person_folder])
    
    print(f"✅ Face embeddings database ready! ({len(embeddings_array)} persons)")
else:
    print(f"⚠ Warning: Face database path not found: {db_path}")

# ------------------- Alert Timing -------------------
alert_timers = {}
ALERT_INTERVAL = 5.0
face_recognition_interval = 0

# ------------------- Face Recognition Cache -------------------
face_cache = {}
CACHE_DURATION = 2.0

def recognize_face(face_img, embeddings_array, threshold=0.4):
    """Optimized face recognition with caching"""
    if not embeddings_array:
        return "Unknown"
    
    try:
        if face_img.shape[0] > 160:
            face_img = cv2.resize(face_img, (160, 160))
        
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        emb = DeepFace.represent(
            rgb_face,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend='opencv'
        )[0]["embedding"]
        emb = np.array(emb)
        
    except Exception as e:
        return "Unknown"
    
    min_dist = float("inf")
    identity = "Unknown"
    
    for name, db_embs_array in embeddings_array.items():
        distances = np.array([cosine(emb, db_emb) for db_emb in db_embs_array])
        min_person_dist = np.min(distances)
        
        if min_person_dist < min_dist and min_person_dist < threshold:
            min_dist = min_person_dist
            identity = name
    
    return identity

# ------------------- Camera Initialization -------------------
def initialize_camera(camera_index=0, backend=cv2.CAP_DSHOW):
    """Initialize camera with fallback options"""
    print(f"Attempting to open camera {camera_index}...")
    
    # Try with DirectShow backend (more reliable on Windows)
    cap = cv2.VideoCapture(camera_index, backend)
    
    if not cap.isOpened():
        print(f"Failed with backend {backend}, trying default backend...")
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Failed with camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)
    
    if cap.isOpened():
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test read
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera initialized successfully!")
            return cap
        else:
            print("⚠ Camera opened but cannot read frames")
            cap.release()
            return None
    else:
        print("❌ Failed to open any camera")
        return None

# ------------------- Process Frame (Optimized) -------------------
def process_frame(frame):
    global safe_zone, alert_timers, face_recognition_interval
    
    if frame is None:
        return None
    
    start_time = cv2.getTickCount()
    
    # Optimize frame size
    height, width = frame.shape[:2]
    if width > 1280:
        scale = 1280 / width
        frame = cv2.resize(frame, (1280, int(height * scale)))
    
    # Run YOLO
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
        device='0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    )[0]
    
    person_centers = []
    total_persons = 0
    persons_outside_zone = 0
    current_time = time.time()
    total_detections = 0
    
    face_recognition_interval += 1
    do_face_recognition = (face_recognition_interval % 15 == 0)
    
    # Draw Safe Zone
    if safe_zone:
        cv2.polylines(frame, [np.array(safe_zone, np.int32)], True, (0,255,255), 3)
    
    # Store all detections grouped by type
    all_detections = []
    persons = []
    ppe_items = []
    
    # ========== COLLECT ALL DETECTIONS ==========
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        total_detections += 1
        
        # Store detection info
        detection = {
            'label': label,
            'conf': conf,
            'bbox': (x1, y1, x2, y2),
            'cls_id': cls_id
        }
        all_detections.append(detection)
        
        if label == "person":
            persons.append((x1, y1, x2, y2, conf))
        elif label in PPE_CLASSES:
            ppe_items.append((x1, y1, x2, y2, label, conf))
    
    # ========== DRAW ALL DETECTIONS ==========
    for detection in all_detections:
        label = detection['label']
        conf = detection['conf']
        x1, y1, x2, y2 = detection['bbox']
        
        # Get color for this detection type
        color = DETECTION_COLORS.get(label, DETECTION_COLORS['default'])
        
        # Determine box thickness (persons and PPE get thicker boxes)
        thickness = 3 if label in ['person'] + PPE_CLASSES else 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Create label with confidence
        label_text = f"{label} {conf:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw filled rectangle as background for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Draw text label
        cv2.putText(
            frame,
            label_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # Black text
            2
        )
    
    # ========== PROCESS PERSONS (for additional features) ==========
    for x1, y1, x2, y2, conf in persons:
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
        
        total_persons += 1
        person_centers.append((cx, cy))
        
        # Check safe zone
        if safe_zone:
            inside = cv2.pointPolygonTest(np.array(safe_zone, np.int32), (cx, cy), False)
            if inside < 0:
                persons_outside_zone += 1
                # Draw warning overlay
                cv2.putText(frame, "⚠ ZONE VIOLATION", (x1, y1 - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
        
        # Face Recognition
        name = "Unknown"
        if do_face_recognition and embeddings_array:
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size > 0:
                cache_key = (x1, y1, x2, y2)
                cached = face_cache.get(cache_key)
                if cached and (current_time - cached[1]) < CACHE_DURATION:
                    name = cached[0]
                else:
                    name = recognize_face(person_roi, embeddings_array)
                    face_cache[cache_key] = (name, current_time)
        
        # Draw name with background below person box
        if name != "Unknown" or embeddings_array:
            name_text = name
            (name_width, name_height), _ = cv2.getTextSize(
                name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            name_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(
                frame,
                (x1, y2 + 5),
                (x1 + name_width + 10, y2 + name_height + 15),
                name_color,
                -1
            )
            cv2.putText(frame, name_text, (x1 + 5, y2 + name_height + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_offset = y2 + name_height + 20
        else:
            y_offset = y2 + 5
        
        # ========== PPE Detection for this person ==========
        wearing_ppe = False
        detected_ppe = {}
        
        for px1, py1, px2, py2, ppe_label, ppe_conf in ppe_items:
            # Check overlap
            overlap_x = max(0, min(x2, px2) - max(x1, px1))
            overlap_y = max(0, min(y2, py2) - max(y1, py1))
            overlap_area = overlap_x * overlap_y
            
            ppe_area = (px2 - px1) * (py2 - py1)
            person_area = (x2 - x1) * (y2 - y1)
            
            if overlap_area > 0.2 * ppe_area or overlap_area > 0.1 * person_area:
                wearing_ppe = True
                if ppe_label not in detected_ppe:
                    detected_ppe[ppe_label] = ppe_conf
                else:
                    detected_ppe[ppe_label] = max(detected_ppe[ppe_label], ppe_conf)
        
        # Display detected PPE
        if detected_ppe:
            ppe_text = ", ".join([f"{label}" for label in detected_ppe.keys()])
            (ppe_width, ppe_height), _ = cv2.getTextSize(
                f"PPE: {ppe_text}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                frame,
                (x1, y_offset),
                (x1 + ppe_width + 10, y_offset + ppe_height + 10),
                (0, 255, 0),
                -1
            )
            cv2.putText(frame, f"PPE: {ppe_text}", (x1 + 5, y_offset + ppe_height + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            y_offset += ppe_height + 15
        
        # ========== Alert Logic ==========
        if name != "Unknown" and not wearing_ppe:
            last = alert_timers.get(name, 0)
            if current_time - last > ALERT_INTERVAL:
                speak(f"{name}, wear your personal protective equipment")
                alert_timers[name] = current_time
            
            # Draw NO PPE warning
            cv2.rectangle(
                frame,
                (x1, y_offset),
                (x1 + 120, y_offset + 30),
                (0, 0, 255),
                -1
            )
            cv2.putText(frame, "⚠ NO PPE!", (x1 + 5, y_offset + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ========== Distance calculation ==========
    for i in range(len(person_centers)):
        for j in range(i+1, len(person_centers)):
            x1, y1 = person_centers[i]
            x2, y2 = person_centers[j]
            dist = math.hypot(x2 - x1, y2 - y1)
            meters = dist / PIXELS_PER_METER
            
            if meters < 5.0:
                # Draw line
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw distance text with background
                mid = ((x1 + x2)//2, (y1 + y2)//2 - 10)
                dist_text = f"{meters:.2f}m"
                (dist_width, dist_height), _ = cv2.getTextSize(
                    dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(
                    frame,
                    (mid[0] - 5, mid[1] - dist_height - 5),
                    (mid[0] + dist_width + 5, mid[1] + 5),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(frame, dist_text, mid,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Calculate FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
    
    # ========== Display enhanced stats ==========
    # Create semi-transparent overlay for stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (320, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"Total Detections: {total_detections}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Persons: {total_persons}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"PPE Items: {len(ppe_items)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    cv2.putText(frame, f"Outside Zone: {persons_outside_zone}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # ========== Detection Summary (bottom right) ==========
    if total_detections > 0:
        # Count detections by type
        detection_counts = {}
        for det in all_detections:
            label = det['label']
            detection_counts[label] = detection_counts.get(label, 0) + 1
        
        # Display summary
        summary_y = height - 20
        summary_x = width - 250
        
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (summary_x - 10, summary_y - (len(detection_counts) * 30) - 10), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        
        for idx, (label, count) in enumerate(sorted(detection_counts.items())):
            color = DETECTION_COLORS.get(label, DETECTION_COLORS['default'])
            y_pos = summary_y - (len(detection_counts) - idx - 1) * 30
            
            # Draw colored square
            cv2.rectangle(frame, (summary_x, y_pos - 15), (summary_x + 15, y_pos), color, -1)
            cv2.rectangle(frame, (summary_x, y_pos - 15), (summary_x + 15, y_pos), (255, 255, 255), 1)
            
            # Draw text
            cv2.putText(frame, f"{label}: {count}", (summary_x + 25, y_pos - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Clean cache
    keys_to_remove = [k for k, (n, t) in face_cache.items() if current_time - t > CACHE_DURATION]
    for k in keys_to_remove:
        del face_cache[k]
    
    return frame

# ------------------- Draw Safe Zone -------------------
drawing = False

def set_safe_zone(event, x, y, flags, param):
    global drawing, safe_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        safe_zone.append((x, y))
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if safe_zone:
            safe_zone[-1] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def define_safe_zone():
    global safe_zone
    safe_zone = []
    
    cap = initialize_camera(backend=cv2.CAP_DSHOW)
    
    if cap is None:
        print("❌ Cannot open camera for safe zone definition")
        return False
    
    cv2.namedWindow("Draw Safe Zone")
    cv2.setMouseCallback("Draw Safe Zone", set_safe_zone)
    
    print("\n📝 Instructions:")
    print("  • Click to draw safe zone points")
    print("  • Press ENTER when done")
    print("  • Press ESC to skip safe zone\n")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            frame_count += 1
            if frame_count > 30:
                print("❌ Camera stopped responding")
                break
            time.sleep(0.1)
            continue
        
        frame_count = 0
        
        if safe_zone:
            cv2.polylines(frame, [np.array(safe_zone, np.int32)], False, (0, 255, 255), 3)
            for pt in safe_zone:
                cv2.circle(frame, pt, 6, (0, 255, 255), -1)
                cv2.circle(frame, pt, 6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Click points | ENTER=Done | ESC=Skip", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Draw Safe Zone", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            print(f"✅ Safe zone defined with {len(safe_zone)} points")
            break
        elif key == 27:  # ESC
            safe_zone = []
            print("⚠ Safe zone skipped")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)
    return True

# ------------------- Webcam Loop -------------------
def run_webcam():
    global alert_timers
    alert_timers.clear()
    
    cap = initialize_camera(backend=cv2.CAP_DSHOW)
    
    if cap is None:
        print("\n❌ ERROR: Cannot access camera!")
        print("\nTroubleshooting steps:")
        print("1. Close all other apps using the camera (Teams, Zoom, etc.)")
        print("2. Check Camera privacy settings in Windows")
        print("3. Try restarting your computer")
        return
    
    print("\n✅ Starting PPE Detection...")
    print("Press 'Q' to quit\n")
    
    cv2.namedWindow("PPE Detection System", cv2.WINDOW_NORMAL)
    
    consecutive_failures = 0
    max_failures = 30
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                print("❌ Camera disconnected or stopped responding")
                break
            time.sleep(0.1)
            continue
        
        consecutive_failures = 0
        
        processed_frame = process_frame(frame)
        
        if processed_frame is not None:
            cv2.imshow("PPE Detection System", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n👋 Exiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released")

def start_webcam():
    success = define_safe_zone()
    if success:
        threading.Thread(target=run_webcam, daemon=True).start()
    else:
        print("\n⚠ Cannot start detection without camera access")

# ------------------- Tkinter GUI -------------------
root = tk.Tk()
root.title("PPE Detection System")
root.geometry("450x310")
root.configure(bg="#2c3e50")

title_frame = tk.Frame(root, bg="#34495e", pady=15)
title_frame.pack(fill="x")

lbl = tk.Label(title_frame, text="🛡️ PPE Detection & Monitoring", 
               font=("Arial", 18, "bold"), bg="#34495e", fg="white")
lbl.pack()

btn_frame = tk.Frame(root, bg="#2c3e50", pady=20)
btn_frame.pack()

btn1 = tk.Button(btn_frame, text="▶ Start Live Detection", command=start_webcam,
                 width=25, height=2, bg="#27ae60", fg="white", 
                 font=("Arial", 12, "bold"), cursor="hand2")
btn1.pack(pady=10)

info_lbl = tk.Label(root, text="✓ Shows ALL YOLO detections with bounding boxes\n✓ Draw safe zone on startup\n✓ Press 'Q' to quit detection",
                    font=("Arial", 9), bg="#2c3e50", fg="#95a5a6")
info_lbl.pack(pady=10)

# Detection Color Legend
legend_frame = tk.Frame(root, bg="#2c3e50")
legend_frame.pack(pady=5)

legend_title = tk.Label(legend_frame, text="Detection Colors:", 
                       font=("Arial", 9, "bold"), bg="#2c3e50", fg="white")
legend_title.pack()

legend_text = "🟢 Person  🟡 Helmet  🟣 Mask  🟠 Vest  🔵 Gloves  🔷 Others"
legend_colors = tk.Label(legend_frame, text=legend_text,
                        font=("Arial", 8), bg="#2c3e50", fg="#95a5a6")
legend_colors.pack()

status_lbl = tk.Label(root, text=f"📊 Face Database: {len(embeddings_array)} persons loaded",
                      font=("Arial", 9), bg="#2c3e50", fg="#3498db")
status_lbl.pack()

print("\n" + "="*50)
print("🚀 PPE Detection System Ready!")
print("="*50 + "\n")

root.mainloop()