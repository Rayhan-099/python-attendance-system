import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import hashlib
from pathlib import Path



#colors

BG_COLOR = "#1c1c1c"
ACCENT_COLOR = "#00fcca"
TEXT_COLOR = "#ffffff"


class AttendanceApp: