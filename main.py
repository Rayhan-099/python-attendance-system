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

# --- CONFIGURATION ---
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

        # Window Setup
        self.title("NeuralFace Attendance 2.0")
        self.geometry("1100x700")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar (Controls)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="NEURAL FACE", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(pady=30, padx=20)

        self.id_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Student ID")
        self.id_entry.pack(pady=10, padx=20)
        
        self.name_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Student Name")
        self.name_entry.pack(pady=10, padx=20)

        self.btn_register = ctk.CTkButton(self.sidebar, text="Register New Face", command=self.register_face, fg_color="#2CC985")
        self.btn_register.pack(pady=20, padx=20)

        self.btn_attendance = ctk.CTkButton(self.sidebar, text="Start Attendance", command=self.start_attendance, fg_color="#3B8ED0")
        self.btn_attendance.pack(pady=10, padx=20)
        
        self.btn_stop = ctk.CTkButton(self.sidebar, text="Stop Camera", command=self.stop_camera, fg_color="#C0392B")
        self.btn_stop.pack(pady=10, padx=20)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Ready", text_color="gray")
        self.status_label.pack(side="bottom", pady=20)

        # Main View (Camera Feed)
        self.main_view = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.camera_label = ctk.CTkLabel(self.main_view, text="", width=640, height=480, fg_color="black")
        self.camera_label.pack(pady=10)

    # Placeholder functions to prevent errors
    def register_face(self): pass
    def start_attendance(self): pass
    def stop_camera(self): pass

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()