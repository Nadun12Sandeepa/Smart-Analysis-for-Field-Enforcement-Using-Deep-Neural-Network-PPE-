import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import math
import numpy as np
import platform
import os
from ultralytics import YOLO
import threading

# ------------------- Beep Sound -------------------
def beep():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(1000, 200)
    elif system == "Darwin":
        os.system('afplay /System/Library/Sounds/Glass.aiff')
    else:
        os.system('echo -e "\a"')

# ------------------- Load YOLO model -------------------
model = YOLO("bestFirst.pt")  # <- replace with your trained weights path

# Default Safe Zone (rectangle)
safe_zone = [(200, 150), (1100, 150), (1100, 550), (200, 550)]
PIXELS_PER_METER = 100

# ------------------- Process Frame -------------------
def process_frame(frame):
    global safe_zone
    # Draw Safe Zone
    if safe_zone:
        cv2.polylines(frame, [np.array(safe_zone, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(frame, "Safe Zone", (safe_zone[0][0], safe_zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    results = model(frame)[0]
    person_centers = []
    total_persons, persons_outside_zone = 0, 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        color = (0, 255, 0) if label.lower() == "person" else (128, 128, 128)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        if label.lower() == "person":
            total_persons += 1
            person_centers.append((cx, cy))
            if safe_zone:
                inside = cv2.pointPolygonTest(np.array(safe_zone, dtype=np.int32), (cx, cy), False)
                if inside < 0:
                    persons_outside_zone += 1
                    beep()

    cv2.putText(frame, f"Total Persons: {total_persons}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Outside Zone: {persons_outside_zone}", (50, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for i in range(len(person_centers)):
        for j in range(i + 1, len(person_centers)):
            x1, y1 = person_centers[i]
            x2, y2 = person_centers[j]
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            distance_m = pixel_distance / PIXELS_PER_METER
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            mid = ((x1 + x2) // 2, (y1 + y2) // 2 - 10)
            cv2.putText(frame, f"{distance_m:.2f} m", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# ------------------- Run Live Webcam -------------------
def run_webcam():
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

# ------------------- Run Video -------------------
def run_video(filepath):
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# ------------------- Run Image -------------------
def run_image(filepath):
    frame = cv2.imread(filepath)
    if frame is None:
        messagebox.showerror("Error", "Could not load image!")
        return
    frame = process_frame(frame)
    cv2.imshow("Image Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------- Tkinter GUI -------------------
def start_webcam():
    threading.Thread(target=run_webcam, daemon=True).start()

def start_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if filepath:
        threading.Thread(target=run_video, args=(filepath,), daemon=True).start()

def start_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if filepath:
        threading.Thread(target=run_image, args=(filepath,), daemon=True).start()

# ------------------- Safe Zone Setup -------------------
def setup_safezone():
    def save_zone():
        try:
            x1, y1 = int(entry1.get()), int(entry2.get())
            x2, y2 = int(entry3.get()), int(entry4.get())
            x3, y3 = int(entry5.get()), int(entry6.get())
            x4, y4 = int(entry7.get()), int(entry8.get())
            global safe_zone
            safe_zone = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            messagebox.showinfo("Safe Zone", f"Safe Zone Updated:\n{safe_zone}")
            win.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer coordinates!")

    win = tk.Toplevel(root)
    win.title("Setup Safe Zone")
    win.geometry("300x300")

    tk.Label(win, text="Enter 4 Points (x,y)").pack(pady=5)

    frame = tk.Frame(win)
    frame.pack()

    entry1 = tk.Entry(frame, width=5); entry2 = tk.Entry(frame, width=5)
    entry3 = tk.Entry(frame, width=5); entry4 = tk.Entry(frame, width=5)
    entry5 = tk.Entry(frame, width=5); entry6 = tk.Entry(frame, width=5)
    entry7 = tk.Entry(frame, width=5); entry8 = tk.Entry(frame, width=5)

    entry1.grid(row=0, column=0); entry2.grid(row=0, column=1)
    entry3.grid(row=1, column=0); entry4.grid(row=1, column=1)
    entry5.grid(row=2, column=0); entry6.grid(row=2, column=1)
    entry7.grid(row=3, column=0); entry8.grid(row=3, column=1)

    tk.Button(win, text="Save Zone", command=save_zone).pack(pady=10)

# ------------------- Main GUI -------------------
root = tk.Tk()
root.title("YOLO Safe Zone Detection")
root.geometry("400x400")

lbl = tk.Label(root, text="YOLOv8 Safe Zone Detection", font=("Arial", 14))
lbl.pack(pady=20)

btn1 = tk.Button(root, text="Start Live Webcam", command=start_webcam, width=25, height=2)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="Detect from Video", command=start_video, width=25, height=2)
btn2.pack(pady=10)

btn3 = tk.Button(root, text="Detect from Image", command=start_image, width=25, height=2)
btn3.pack(pady=10)

btn4 = tk.Button(root, text="Setup Safe Zone", command=setup_safezone, width=25, height=2, bg="lightblue")
btn4.pack(pady=15)

root.mainloop()
