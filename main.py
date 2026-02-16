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

        # UI Components
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

        # State Variables
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
        else:
            self.update_status("No database found.")

    def register_face(self):
        s_id = self.id_entry.get()
        name = self.name_entry.get()

        if not s_id or not name:
            self.update_status("Error: Enter ID and Name!")
            return

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.update_status("Error: Camera failed.")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        
        if not boxes:
            self.update_status("No face detected.")
            return
        
        encodings = face_recognition.face_encodings(rgb_frame, boxes)
        if encodings:
            self.known_encodings.append(encodings[0])
            self.known_names.append(f"{name}_{s_id}")
            data = {"encodings": self.known_encodings, "names": self.known_names}
            with open(ENCODINGS_FILE, "wb") as f:
                f.write(pickle.dumps(data))
            self.update_status(f"Registered: {name}")

    def start_attendance(self): pass
    def stop_camera(self): pass

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()