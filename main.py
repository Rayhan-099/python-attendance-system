import customtkinter as ctk
import cv2
import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime
import csv
import numpy as np
from PIL import Image, ImageTk

DATA_DIR = "data"
ENCODINGS_FILE = os.path.join(DATA_DIR, "encodings.pickle")
ATTENDANCE_FILE = f"Attendance_{datetime.now().strftime('%d-%m-%Y')}.csv"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NeuralFace Attendance 2.0")
        self.geometry("1100x700")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="NEURAL FACE", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=30)
        
        self.id_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Student ID")
        self.id_entry.pack(pady=10, padx=20)
        self.name_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Student Name")
        self.name_entry.pack(pady=10, padx=20)

        ctk.CTkButton(self.sidebar, text="Register New Face", command=self.register_face, fg_color="#2CC985").pack(pady=20, padx=20)
        ctk.CTkButton(self.sidebar, text="Start Attendance", command=self.start_attendance, fg_color="#3B8ED0").pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="Stop Camera", command=self.stop_camera, fg_color="#C0392B").pack(pady=10, padx=20)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Ready", text_color="gray")
        self.status_label.pack(side="bottom", pady=20)

        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.camera_label = ctk.CTkLabel(self.main_view, text="", width=640, height=480, fg_color="black")
        self.camera_label.pack(pady=10)

        self.cap = None
        self.is_running = False
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()

    def update_status(self, text):
        self.status_label.configure(text=f"Status: {text}")

    def load_encodings(self):
        if os.path.exists(ENCODINGS_FILE):
            try:
                data = pickle.loads(open(ENCODINGS_FILE, "rb").read())
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
                self.update_status(f"Loaded {len(self.known_names)} users.")
            except:
                self.update_status("Error loading data.")

    def register_face(self):
        s_id = self.id_entry.get()
        name = self.name_entry.get()
        if not s_id or not name:
            self.update_status("Error: Enter ID and Name!")
            return

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame)
            if boxes:
                encodings = face_recognition.face_encodings(rgb_frame, boxes)
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(f"{name}_{s_id}")
                    data = {"encodings": self.known_encodings, "names": self.known_names}
                    with open(ENCODINGS_FILE, "wb") as f:
                        f.write(pickle.dumps(data))
                    self.update_status(f"Registered: {name}")
                    self.id_entry.delete(0, 'end')
                    self.name_entry.delete(0, 'end')

    def mark_attendance(self, name):
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        file_exists = os.path.isfile(ATTENDANCE_FILE)
        
        if file_exists:
            df = pd.read_csv(ATTENDANCE_FILE)
            if name in df['Name'].values:
                return

        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Time'])
            writer.writerow([name, time_str])
        self.update_status(f"Marked: {name}")

    def start_attendance(self):
        if self.is_running: return
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def stop_camera(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.camera_label.configure(image="")
        self.update_status("Camera Stopped")

    def update_camera(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if ret:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        self.mark_attendance(name)

                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.camera_label.configure(image=imgtk)
            self.camera_label.image = imgtk

        self.after(10, self.update_camera)

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()