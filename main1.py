import streamlit as st
import cv2
import pandas as pd
import numpy as np
import sqlite3
import os
import time
import threading
import smtplib
import calendar
import queue
import logging
import base64
import json
from datetime import datetime, timedelta, date
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ==========================================
# 0. SYSTEM INITIALIZATION & LOGGING
# ==========================================
# Configure Streamlit page settings
st.set_page_config(
    page_title="Sentinel AI | Titan Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Setup System Logging
logging.basicConfig(
    filename='system_core.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------
# GLOBAL CONSTANTS & FILE PATHS
# ------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sentinel_titan.db")
FACES_DATA_DIR = os.path.join(BASE_DIR, "biometric_vault")
UNKNOWN_LOGS_DIR = os.path.join(BASE_DIR, "security_breaches")
TRAINER_PATH = os.path.join(BASE_DIR, "neural_model.yml")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

# Ensure all critical system directories exist
for folder in [FACES_DATA_DIR, UNKNOWN_LOGS_DIR, ASSETS_DIR, EXPORTS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Created system directory: {folder}")

# Initialize Audio Engine
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except ImportError:
    logging.warning("Pygame not installed. Audio feedback disabled.")
    AUDIO_AVAILABLE = False

# ------------------------------------------
# SESSION STATE MANAGEMENT
# ------------------------------------------
# We use session state to persist data across Streamlit reruns
if 'theme' not in st.session_state: st.session_state.theme = 'dark'
if 'page' not in st.session_state: st.session_state.page = 'kiosk'
if 'cam_index' not in st.session_state: st.session_state.cam_index = 0
if 'admin_auth' not in st.session_state: st.session_state.admin_auth = False
if 'last_log_time' not in st.session_state: st.session_state.last_log_time = {}
if 'camera_active' not in st.session_state: st.session_state.camera_active = False
if 'last_email_time' not in st.session_state: st.session_state.last_email_time = datetime.min
if 'security_queue' not in st.session_state: st.session_state.security_queue = queue.Queue()
if 'sync_status' not in st.session_state: st.session_state.sync_status = "System Ready"

# ==========================================
# 1. ADVANCED UI ENGINE (TITAN GLASS)
# ==========================================
def inject_titan_ui():
    """
    Injects high-end CSS to override Streamlit defaults.
    Creates a translucent, frosted-glass look with neon accents.
    """
    
    # 1. Define Color Palettes based on Theme
    if st.session_state.theme == 'dark':
        # Dark Cyberpunk Palette
        bg_image = "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070" 
        glass_bg = "rgba(16, 20, 28, 0.85)"
        glass_border = "1px solid rgba(0, 242, 254, 0.15)"
        text_color = "#e2e8f0"
        accent_color = "#00f2fe" # Neon Cyan
        accent_gradient = "linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)"
        sidebar_bg = "rgba(10, 15, 20, 0.9)"
        shadow_light = "0 8px 32px 0 rgba(0, 0, 0, 0.6)"
        success_color = "#00ff9d"
        danger_color = "#ff0055"
        warning_color = "#ffdd00"
    else:
        # Light Corporate Palette
        bg_image = "https://images.unsplash.com/photo-1497366216548-37526070297c?q=80&w=2069" 
        glass_bg = "rgba(255, 255, 255, 0.75)"
        glass_border = "1px solid rgba(255, 255, 255, 0.6)"
        text_color = "#1e293b"
        accent_color = "#2563eb" # Royal Blue
        accent_gradient = "linear-gradient(135deg, #2563eb 0%, #4f46e5 100%)"
        sidebar_bg = "rgba(255, 255, 255, 0.9)"
        shadow_light = "0 8px 32px 0 rgba(31, 38, 135, 0.15)"
        success_color = "#10b981"
        danger_color = "#ef4444"
        warning_color = "#f59e0b"

    # 2. Inject CSS
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

        /* GLOBAL RESET */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: {text_color};
            scroll-behavior: smooth;
        }}
        
        /* BACKGROUND */
        /* BACKGROUND */
        .stApp {{
            /* We add a linear-gradient overlay of black with 70% opacity */
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('{bg_image}');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        
        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            backdrop-filter: blur(20px);
            border-right: {glass_border};
            box-shadow: 10px 0 30px rgba(0,0,0,0.3);
        }}

        /* GLASS CARD COMPONENT */
        .glass-card {{
            background: {glass_bg};
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border: {glass_border};
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: {shadow_light};
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        .glass-card::before {{
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
            transition: 0.5s;
        }}
        .glass-card:hover::before {{
            left: 100%;
        }}
        .glass-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
            border-color: {accent_color};
        }}

        /* KPI METRICS */
        .metric-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            transition: 0.3s;
        }}
        .metric-container:hover {{
            background: rgba(255,255,255,0.07);
        }}
        .metric-label {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            opacity: 0.7;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 2.8rem;
            font-weight: 800;
            background: {accent_gradient};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* BUTTONS */
        .stButton>button {{
            background: {accent_gradient};
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 1.5rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            width: 100%;
            box-shadow: 0 4px 15px {accent_color}40;
        }}
        .stButton>button:hover {{
            box-shadow: 0 8px 25px {accent_color}60;
            transform: translateY(-2px);
        }}
        .stButton>button:active {{
            transform: translateY(1px);
        }}
        
        /* STATUS INDICATORS */
        .status-badge {{
            padding: 5px 10px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 0.8rem;
            display: inline-block;
        }}
        .status-success {{ background: {success_color}20; color: {success_color}; border: 1px solid {success_color}40; }}
        .status-danger {{ background: {danger_color}20; color: {danger_color}; border: 1px solid {danger_color}40; }}
        .status-warning {{ background: {warning_color}20; color: {warning_color}; border: 1px solid {warning_color}40; }}

        /* INPUT FIELDS */
        .stTextInput>div>div, .stSelectbox>div>div, .stNumberInput>div>div, .stDateInput>div>div {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: {text_color} !important;
            border-radius: 12px;
            border: {glass_border};
            transition: 0.3s;
        }}
        .stTextInput>div>div:focus-within {{
            border-color: {accent_color};
            box-shadow: 0 0 15px {accent_color}30;
        }}

        /* DATAFRAMES */
        [data-testid="stDataFrame"] {{
            background: transparent !important;
        }}
        
        /* HEADERS & TYPOGRAPHY */
        h1, h2, h3 {{
            font-weight: 800;
            letter-spacing: -0.5px;
        }}
        
        /* ANIMATIONS */
        @keyframes pulse-ring {{
            0% {{ transform: scale(0.33); opacity: 0.5; }}
            80%, 100% {{ opacity: 0; }}
        }}
        .live-indicator {{
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            background: {danger_color};
            box-shadow: 0 0 10px {danger_color};
            margin-right: 8px;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATABASE ENGINE
# ==========================================
class DatabaseManager:
    """
    Handles all SQLite operations with error handling and logging.
    Thread-safe implementation for Streamlit.
    """
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.init_structure()

    def init_structure(self):
        """Initializes all necessary tables with Schema Migration logic."""
        try:
            # 1. Users Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    dept TEXT,
                    role TEXT,
                    email TEXT,
                    phone TEXT,
                    face_id INTEGER UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'Active'
                )
            """)
            
            # 2. Attendance Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    date TEXT,
                    time TEXT,
                    type TEXT,
                    status TEXT,
                    confidence REAL,
                    synced INTEGER DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            # 3. Settings Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # 4. Holidays Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS holidays (
                    date TEXT PRIMARY KEY,
                    description TEXT,
                    type TEXT DEFAULT 'Public'
                )
            """)
            
            # Populate Defaults
            defaults = {
                'admin_pwd': 'admin',
                'system_mode': 'Corporate', # or 'Education'
                'shift_start': '09:00',
                'shift_end': '17:00',
                'org_name': 'Sentinel Inc.',
                'attendance_threshold': '75',
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': '587',
                'sender_email': '',
                'sender_password': '',
                'receiver_email': ''
            }
            
            for k, v in defaults.items():
                self.cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
                
            self.conn.commit()
            logging.info("Database structure initialized successfully.")
            
        except Exception as e:
            logging.error(f"Database Init Error: {e}")
            st.error(f"Critical Database Error: {e}")

    # --- CONFIGURATION METHODS ---
    def get_config(self, key):
        try:
            res = self.cursor.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return res[0] if res else None
        except Exception as e:
            logging.error(f"Config Read Error ({key}): {e}")
            return None

    def set_config(self, key, value):
        try:
            self.cursor.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Config Write Error ({key}): {e}")
            return False

    # --- LOGGING METHODS ---
    def log_attendance(self, user_id, type_, status, conf):
        now = datetime.now()
        try:
            self.cursor.execute("""
                INSERT INTO attendance (user_id, date, time, type, status, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), type_, status, conf))
            self.conn.commit()
            logging.info(f"Attendance Logged: {user_id} - {status}")
            return True
        except Exception as e:
            logging.error(f"Attendance Log Error: {e}")
            return False
    
    def manual_log(self, user_id, date_str, status):
        """Allows admin to overwrite attendance."""
        try:
            # Delete existing manual entries for that day to avoid dupes
            self.cursor.execute("DELETE FROM attendance WHERE user_id=? AND date=? AND type='Manual'", (user_id, date_str))
            
            self.cursor.execute("""
                INSERT INTO attendance (user_id, date, time, type, status, confidence)
                VALUES (?, ?, '09:00:00', 'Manual', ?, 100.0)
            """, (user_id, date_str, status))
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Manual Log Error: {e}")
            return False

    # --- USER METHODS ---
    def add_user(self, uid, name, dept, role, email, face_id):
        try:
            self.cursor.execute("INSERT INTO users (id, name, dept, role, email, face_id) VALUES (?,?,?,?,?,?)",
                               (uid, name, dept, role, email, face_id))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            logging.error(f"Add User Error: {e}")
            return False

    def delete_user(self, uid):
        try:
            self.cursor.execute("DELETE FROM users WHERE id=?", (uid,))
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Delete User Error: {e}")
            return False

db = DatabaseManager()

# ==========================================
# 3. BACKGROUND SECURITY ENGINE
# ==========================================
class SecurityEngine:
    """
    Handles background tasks: Email alerts, Cloud Sync, etc.
    Runs in separate threads to ensure UI responsiveness.
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Background worker consuming the alert queue."""
        while True:
            try:
                task = self.queue.get()
                if task is None: break
                
                task_type = task.get('type')
                if task_type == 'email':
                    self._send_email(task['path'], task['time'])
                
                self.queue.task_done()
            except Exception as e:
                logging.error(f"Worker Error: {e}")

    def trigger_breach_alert(self, image_path, timestamp):
        """Public method to queue an alert."""
        self.queue.put({'type': 'email', 'path': image_path, 'time': timestamp})

    def _send_email(self, image_path, timestamp):
        sender = db.get_config('sender_email')
        password = db.get_config('sender_password')
        receiver = db.get_config('receiver_email')
        server = db.get_config('smtp_server')
        port = db.get_config('smtp_port')
        
        if not sender or not password or not receiver:
            logging.warning("Email credentials missing. Skipping alert.")
            return

        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"🚨 SENTINEL ALERT: Intruder at {timestamp}"
            msg['From'] = sender
            msg['To'] = receiver
            
            body = f"""
            SECURITY BREACH DETECTED
            ------------------------
            System: Sentinel AI Titan
            Time: {timestamp}
            Location: Main Entry (Cam Index {st.session_state.cam_index})
            
            An unidentified individual was detected. See attached evidence.
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach Image
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)
            
            # SMTP Connection
            with smtplib.SMTP(server, int(port)) as s:
                s.starttls()
                s.login(sender, password)
                s.send_message(msg)
                logging.info(f"Security Alert sent to {receiver}")
                
        except Exception as e:
            logging.error(f"Email Send Failed: {e}")

security_service = SecurityEngine()

# ==========================================
# 4. BIOMETRIC NEURAL ENGINE
# ==========================================
class FaceEngine:
    """
    Wraps OpenCV's LBPH Face Recognizer.
    Handles Training, Saving, and Prediction.
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_loaded = False
        self.load_weights()

    def load_weights(self):
        if os.path.exists(TRAINER_PATH):
            try:
                self.recognizer.read(TRAINER_PATH)
                self.model_loaded = True
                logging.info("Neural Weights Loaded.")
            except Exception as e:
                logging.error(f"Model Load Failed: {e}")
                self.model_loaded = False

    def train_model(self):
        """
        Scans the vault, prepares histograms, and trains the model.
        Returns (Success: bool, Message: str)
        """
        faces = []
        ids = []
        
        if not os.path.exists(FACES_DATA_DIR):
            return False, "Biometric Vault Missing"

        try:
            user_dirs = os.listdir(FACES_DATA_DIR)
            if not user_dirs:
                return False, "Vault is empty"

            for folder in user_dirs:
                path = os.path.join(FACES_DATA_DIR, folder)
                if not os.path.isdir(path): continue
                
                try:
                    face_id = int(folder)
                except ValueError: continue

                for img_name in os.listdir(path):
                    if img_name.startswith('.'): continue
                    img_path = os.path.join(path, img_name)
                    
                    # Convert to Grayscale
                    img_pil = Image.open(img_path).convert('L') 
                    img_numpy = np.array(img_pil, 'uint8')
                    
                    faces.append(img_numpy)
                    ids.append(face_id)
            
            if len(faces) > 0:
                self.recognizer.train(faces, np.array(ids))
                self.recognizer.save(TRAINER_PATH)
                self.model_loaded = True
                return True, f"Model Retrained: {len(faces)} samples processed."
            else:
                return False, "No valid face data found."
                
        except Exception as e:
            logging.error(f"Training Error: {e}")
            return False, str(e)

ai_engine = FaceEngine()

# ==========================================
# 5. UTILITY FUNCTIONS
# ==========================================
def play_sound(sound_type="success"):
    if not AUDIO_AVAILABLE: return
    try:
        filename = "success.wav" if sound_type == "success" else "alert.wav"
        path = os.path.join(ASSETS_DIR, filename)
        if os.path.exists(path):
            pygame.mixer.Sound(path).play()
    except Exception as e:
        logging.error(f"Audio Error: {e}")

def get_labels():
    """Returns dynamic labels based on system mode (Corp vs Edu)."""
    mode = db.get_config('system_mode')
    if mode == 'Education':
        return {
            'usr': 'Student', 
            'usrs': 'Students',
            'id': 'Roll No', 
            'dept': 'Class', 
            'role': 'Section'
        }
    return {
        'usr': 'Employee', 
        'usrs': 'Employees',
        'id': 'Emp ID', 
        'dept': 'Department', 
        'role': 'Designation'
    }

# ==========================================
# 6. PAGE: KIOSK DASHBOARD (THE CORE)
# ==========================================
def render_kiosk():
    labels = get_labels()
    org_name = db.get_config('org_name')
    
    # 1. HEADER SECTION
    col_logo, col_clock = st.columns([3, 1])
    with col_logo:
        st.markdown(f"""
        <h1 style='font-size: 3.5rem; margin-bottom: 0;'>
            {org_name} 
            <span style='font-size: 1.5rem; color: #00f2fe; vertical-align: middle;'>| {labels['usr']} Access</span>
        </h1>
        <div style='display: flex; align-items: center; margin-top: -10px;'>
            <div class='live-indicator'></div>
            <span style='opacity: 0.7; font-family: monospace;'>SYSTEM ONLINE • SECURE MODE</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_clock:
        st.markdown(f"""
        <div class='glass-card' style='padding: 15px; text-align: right;'>
            <div style='font-size: 2.2rem; font-weight: 800; font-family: "JetBrains Mono"; color: #00f2fe;'>
                {datetime.now().strftime('%H:%M')}
            </div>
            <div style='font-size: 0.9rem; opacity: 0.7; letter-spacing: 1px;'>
                {datetime.now().strftime('%A, %d %B %Y').upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. MAIN INTERFACE GRID
    col_camera, col_info = st.columns([1.8, 1])
    
    with col_camera:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        # Create a placeholder for the video feed. This is critical for real-time updates.
        camera_placeholder = st.empty()
        
        # Overlay Controls
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🟢 INITIATE SCANNER", use_container_width=True):
                st.session_state.camera_active = True
        with btn_col2:
            if st.button("🔴 TERMINATE FEED", use_container_width=True):
                st.session_state.camera_active = False
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        # We use placeholders here so we can update them FROM INSIDE the camera loop
        # This fixes the "Live attendance only updates when terminate" issue.
        metrics_placeholder = st.empty()
        logs_placeholder = st.empty()
        
        # Initial State Render
        today_date = datetime.now().strftime("%Y-%m-%d")
        count = db.cursor.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date=?", (today_date,)).fetchone()[0]
        
        metrics_placeholder.markdown(f"""
        <div class='glass-card'>
            <div style='text-align: center;'>
                <div class='metric-label'>TOTAL CHECK-INS</div>
                <div class='metric-value'>{count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        logs_placeholder.markdown(f"""
        <div class='glass-card'>
            <h4 style='border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 10px;'>
                📋 RECENT ACTIVITY
            </h4>
            <div style='text-align: center; opacity: 0.6; padding: 20px;'>
                Waiting for feed...
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. CAMERA PROCESSING LOOP
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(int(st.session_state.cam_index))
        
        # Pre-fetch user map to optimize loop speed (avoid DB hit every frame)
        all_users = db.cursor.execute("SELECT face_id, name, id FROM users").fetchall()
        user_map = {r[0]: {'name': r[1], 'uid': r[2]} for r in all_users}
        
        shift_start = db.get_config('shift_start')
        
        frame_count = 0
        
        while cap.isOpened() and st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera Hardware Error. Please restart the app or check connections.")
                break
            
            frame_count += 1
            
            # 3.1 Face Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = ai_engine.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            
            curr_time = datetime.now()
            
            for (x, y, w, h) in faces:
                name = "UNKNOWN"
                color = (0, 0, 255) # Red
                
                # 3.2 Recognition
                try:
                    if ai_engine.model_loaded:
                        label_id, confidence = ai_engine.recognizer.predict(gray[y:y+h, x:x+w])
                        
                        # Confidence Threshold (Lower is better in LBPH)
                        if confidence < 65 and label_id in user_map:
                            user_data = user_map[label_id]
                            name = user_data['name']
                            uid = user_data['uid']
                            color = (0, 255, 0) # Green
                            
                            # 3.3 Attendance Logic
                            # Debounce mechanism: Prevent multiple logs within 5 minutes
                            last_log = st.session_state.last_log_time.get(uid, datetime.min)
                            if (curr_time - last_log).total_seconds() > 300:
                                
                                # Determine Check-In vs Check-Out
                                existing_logs = db.cursor.execute("SELECT COUNT(*) FROM attendance WHERE user_id=? AND date=?", (uid, today_date)).fetchone()[0]
                                type_ = "Check-In" if existing_logs == 0 else "Check-Out"
                                
                                # Determine Status (Late/OnTime)
                                status = "Present"
                                if type_ == "Check-In":
                                    status = "Late" if curr_time.strftime("%H:%M") > shift_start else "On Time"
                                
                                # Write to DB
                                db.log_attendance(uid, type_, status, confidence)
                                st.session_state.last_log_time[uid] = curr_time
                                play_sound("success")
                                st.toast(f"✅ Verified: {name} ({status})")
                                
                    # 3.4 Security Breach Logic
                    elif name == "UNKNOWN":
                        # Draw Warning
                        cv2.putText(frame, "⚠️ INTRUDER", (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        
                        # Throttle Emails (1 per 60 secs)
                        last_email = st.session_state.last_email_time
                        if (curr_time - last_email).total_seconds() > 60:
                            st.session_state.last_email_time = curr_time
                            
                            # Capture & Save
                            fname = f"breach_{curr_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            fpath = os.path.join(UNKNOWN_LOGS_DIR, fname)
                            cv2.imwrite(fpath, frame)
                            
                            play_sound("alert")
                            st.toast("🚨 Security Breach Detected! Alert Sent.", icon="⚠️")
                            
                            # Trigger Background Email
                            security_service.trigger_breach_alert(fpath, curr_time.strftime("%H:%M:%S"))

                except Exception as e:
                    pass # Prevent crashing on individual face errors
                
                # Draw Visuals
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                # Corner accents
                cv2.line(frame, (x,y), (x+20, y), color, 4)
                cv2.line(frame, (x,y), (x, y+20), color, 4)
                cv2.line(frame, (x+w,y+h), (x+w-20, y+h), color, 4)
                cv2.line(frame, (x+w,y+h), (x+w, y+h-20), color, 4)
                
                # Name Tag
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                cv2.putText(frame, name, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # 3.5 Render Frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # 3.6 REAL-TIME DASHBOARD UPDATE
            # Only update SQL queries every 10 frames to save performance
            if frame_count % 10 == 0:
                # Update Metrics
                curr_count = db.cursor.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date=?", (today_date,)).fetchone()[0]
                metrics_placeholder.markdown(f"""
                <div class='glass-card'>
                    <div style='text-align: center;'>
                        <div class='metric-label'>CHECKED IN TODAY</div>
                        <div class='metric-value'>{curr_count}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Update Logs Table
                recents = db.cursor.execute(f"""
                    SELECT u.name, a.time, a.status 
                    FROM attendance a 
                    JOIN users u ON a.user_id = u.id 
                    WHERE a.date='{today_date}' 
                    ORDER BY a.log_id DESC LIMIT 6
                """).fetchall()
                
                if recents:
                    df_rec = pd.DataFrame(recents, columns=["Name", "Time", "Status"])
                    # Generate HTML table manually for cleaner look inside glass card
                    table_html = df_rec.to_html(index=False, classes='dataframe', border=0)
                    logs_placeholder.markdown(f"""
                    <div class='glass-card'>
                        <h4 style='border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 10px;'>
                            📋 RECENT ACTIVITY
                        </h4>
                        {table_html}
                    </div>
                    """, unsafe_allow_html=True)

            time.sleep(0.01) # Yield to CPU
        
        cap.release()
        camera_placeholder.empty()
        st.session_state.camera_active = False # Reset state on exit

# ==========================================
# 7. PAGE: ADMIN COMMAND CENTER
# ==========================================
def render_admin():
    labels = get_labels()
    
    # 1. ADMIN HEADER
    col_adm_head, col_adm_btn = st.columns([8, 1.5])
    col_adm_head.title("COMMAND CENTER")
    if col_adm_btn.button("🔒 LOGOUT", use_container_width=True):
        st.session_state.admin_auth = False
        st.session_state.page = 'kiosk'
        st.rerun()

    tabs = st.tabs(["📊 ANALYTICS", "👥 USER MANAGEMENT", "📝 REPORTS & FIXES", "⚙️ SETTINGS"])

    # --- TAB 1: ANALYTICS ---
    with tabs[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Filter
        d_sel = st.date_input("Select Analysis Date", date.today())
        d_str = d_sel.strftime("%Y-%m-%d")
        
        # Data Fetch
        df_att = pd.read_sql_query(f"""
            SELECT u.dept, a.status, a.time, a.type 
            FROM attendance a 
            JOIN users u ON a.user_id = u.id 
            WHERE a.date='{d_str}'
        """, db.conn)
        
        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        total_u = db.cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        present_u = df_att[df_att['type']=='Check-In']['status'].count()
        absent_u = total_u - present_u
        late_u = df_att[df_att['status']=='Late'].shape[0]
        
        k1.markdown(f"<div class='glass-card'><div class='metric-label'>TOTAL {labels['usrs'].upper()}</div><div class='metric-value'>{total_u}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='glass-card'><div class='metric-label'>PRESENT</div><div class='metric-value'>{present_u}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='glass-card'><div class='metric-label'>ABSENT</div><div class='metric-value'>{absent_u}</div></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='glass-card'><div class='metric-label'>LATE ARRIVALS</div><div class='metric-value'>{late_u}</div></div>", unsafe_allow_html=True)

        # Charts
        c_ch1, c_ch2 = st.columns(2)
        
        with c_ch1:
            st.markdown("<div class='glass-card'><h4>Arrival Timeline</h4>", unsafe_allow_html=True)
            if not df_att.empty:
                df_att['hour'] = pd.to_datetime(df_att['time'], format='%H:%M:%S').dt.hour
                fig = px.histogram(df_att[df_att['type']=='Check-In'], x='hour', nbins=24, 
                                 template="plotly_dark", color_discrete_sequence=['#00f2fe'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No attendance data for this date.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_ch2:
            st.markdown(f"<div class='glass-card'><h4>{labels['dept']} Distribution</h4>", unsafe_allow_html=True)
            df_dept = pd.read_sql_query(f"""
                SELECT u.dept, COUNT(*) as cnt 
                FROM attendance a JOIN users u ON a.user_id = u.id 
                WHERE a.date='{d_str}' GROUP BY u.dept
            """, db.conn)
            if not df_dept.empty:
                fig2 = px.pie(df_dept, values='cnt', names='dept', hole=0.5, template="plotly_dark")
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Security Gallery
        st.markdown("<div class='glass-card'><h3>🚨 Intruder Gallery (Last 24h)</h3>", unsafe_allow_html=True)
        imgs = sorted(os.listdir(UNKNOWN_LOGS_DIR), reverse=True)[:6]
        if imgs:
            cols = st.columns(6)
            for idx, img in enumerate(imgs):
                with cols[idx]:
                    st.image(os.path.join(UNKNOWN_LOGS_DIR, img), caption=img.split('.')[0], use_container_width=True)
        else:
            st.success("No security breaches detected recently.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: USER MANAGEMENT ---
    with tabs[1]:
        c_add, c_list = st.columns([1.2, 1.8])
        
        with c_add:
            st.markdown(f"<div class='glass-card'><h3>➕ Register {labels['usr']}</h3>", unsafe_allow_html=True)
            with st.form("reg_form"):
                uid = st.text_input(f"{labels['id']} (Unique)")
                name = st.text_input("Full Name")
                dept = st.text_input(labels['dept'])
                role = st.text_input(labels['role'])
                email = st.text_input("Email Address")
                
                if st.form_submit_button("CREATE PROFILE"):
                    try:
                        curr = db.cursor.execute("SELECT MAX(face_id) FROM users").fetchone()[0]
                        fid = 1 if curr is None else curr + 1
                        if db.add_user(uid, name, dept, role, email, fid):
                            st.success(f"User Created. Face ID: {fid}")
                        else:
                            st.error("ID Already Exists.")
                    except Exception as e: st.error(str(e))
            st.markdown("</div>", unsafe_allow_html=True)
            
            # --- VISUAL ENROLLMENT WIDGET (NEW REQUEST) ---
            st.markdown(f"<div class='glass-card'><h3>📸 Face Enrollment</h3>", unsafe_allow_html=True)
            
            all_u = db.cursor.execute("SELECT face_id, name, id FROM users").fetchall()
            target = st.selectbox("Select User to Train", [f"{u[1]} ({u[2]})" for u in all_u])
            
            # Placeholder for the registration camera
            reg_frame = st.empty()
            reg_progress = st.empty()
            
            if st.button("START CAPTURE (50 SAMPLES)"):
                # Extract Face ID
                fid = int(target.split('(')[1].strip(')').strip()) # Simplified extraction
                # Better: lookup
                fid = next((u[0] for u in all_u if str(u[2]) in target), None)
                
                if fid:
                    save_path = os.path.join(FACES_DATA_DIR, str(fid))
                    os.makedirs(save_path, exist_ok=True)
                    
                    cap = cv2.VideoCapture(st.session_state.cam_index)
                    count = 0
                    
                    while count < 50:
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # 1. Show User Frame (so they can center face)
                        disp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 2. Detect & Save
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = ai_engine.face_cascade.detectMultiScale(gray, 1.2, 5)
                        
                        for (x,y,w,h) in faces:
                            count += 1
                            cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), gray[y:y+h, x:x+w])
                            # Draw visual feedback
                            cv2.rectangle(disp_frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        
                        reg_frame.image(disp_frame, caption=f"Capturing: {count}/50", use_container_width=True)
                        reg_progress.progress(count/50)
                        time.sleep(0.05)
                    
                    cap.release()
                    reg_frame.empty()
                    
                    with st.spinner("Updating Neural Network..."):
                        s, m = ai_engine.train_model()
                        if s: st.success("Biometric Profile Updated!")
                        else: st.error(m)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_list:
            st.markdown(f"<div class='glass-card'><h3>📂 {labels['usr']} Directory</h3>", unsafe_allow_html=True)
            udf = pd.read_sql("SELECT * FROM users", db.conn)
            st.dataframe(udf, use_container_width=True, height=600)
            
            del_id = st.text_input(f"Delete {labels['id']}")
            if st.button("DELETE RECORD"):
                if db.delete_user(del_id):
                    st.success("User deleted.")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 3: REPORTS ---
    with tabs[2]:
        st.markdown(f"<div class='glass-card'><h3>🗓️ Attendance Matrix</h3>", unsafe_allow_html=True)
        c_m, c_y, c_b = st.columns([1,1,1])
        mo = c_m.selectbox("Month", range(1,13), index=datetime.now().month-1)
        yr = c_y.number_input("Year", value=datetime.now().year)
        
        if c_b.button("GENERATE REPORT"):
            # Matrix Generation Logic
            days = calendar.monthrange(yr, mo)[1]
            users = db.cursor.execute("SELECT id, name FROM users").fetchall()
            
            matrix = []
            for uid, uname in users:
                row = {"ID": uid, "Name": uname}
                p_days = 0
                for d in range(1, days+1):
                    dt = date(yr, mo, d).strftime("%Y-%m-%d")
                    
                    # Holiday Check
                    hol = db.cursor.execute("SELECT description FROM holidays WHERE date=?", (dt,)).fetchone()
                    
                    # Attendance Check
                    att = db.cursor.execute("SELECT status FROM attendance WHERE user_id=? AND date=?", (uid, dt)).fetchone()
                    
                    code = "A"
                    if hol: code = "H"
                    elif att:
                        s = att[0]
                        if s in ["Present", "On Time", "Late"]: 
                            code = "P"
                            p_days += 1
                        elif s == "Leave": code = "L"
                        elif s == "Work From Home": code = "WFH"
                    
                    row[str(d)] = code
                
                row["%"] = f"{(p_days/days)*100:.1f}"
                matrix.append(row)
            
            df_m = pd.DataFrame(matrix)
            
            # Styling
            def colorize(val):
                c = ''
                if val == 'P': c = 'background-color: rgba(0,255,0,0.2); color: #0f0'
                elif val == 'A': c = 'background-color: rgba(255,0,0,0.2); color: #f00'
                elif val == 'H': c = 'color: orange'
                return c
                
            st.dataframe(df_m.style.applymap(colorize), use_container_width=True)
            
            csv = df_m.to_csv().encode('utf-8')
            st.download_button("📥 Export CSV", csv, "report.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Manual Override
        st.markdown("<div class='glass-card'><h3>🛠️ Corrections</h3>", unsafe_allow_html=True)
        c_fix1, c_fix2 = st.columns(2)
        u_fix = c_fix1.text_input(f"Enter {labels['id']} to Fix")
        d_fix = c_fix2.date_input("Date")
        s_fix = st.selectbox("Status", ["Present", "Leave", "Work From Home", "Absent"])
        
        if st.button("APPLY CORRECTION"):
            db.manual_log(u_fix, d_fix.strftime("%Y-%m-%d"), s_fix)
            st.success("Record updated.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 4: SETTINGS ---
    with tabs[3]:
        st.markdown("<div class='glass-card'><h3>⚙️ Configuration</h3>", unsafe_allow_html=True)
        
        with st.expander("🚨 Security & Email Alerts", expanded=True):
            st.caption("Configure where intruder alerts are sent.")
            c_e1, c_e2 = st.columns(2)
            smtp_s = c_e1.text_input("SMTP Server", value=db.get_config('smtp_server'))
            smtp_p = c_e2.text_input("SMTP Port", value=db.get_config('smtp_port'))
            send_e = c_e1.text_input("Sender Email", value=db.get_config('sender_email'))
            send_p = c_e2.text_input("Sender Password (App Pwd)", type="password", value=db.get_config('sender_password'))
            recv_e = st.text_input("Receiver Email (Admin)", value=db.get_config('receiver_email'))
            
            if st.button("SAVE EMAIL CONFIG"):
                db.set_config('smtp_server', smtp_s)
                db.set_config('smtp_port', smtp_p)
                db.set_config('sender_email', send_e)
                db.set_config('sender_password', send_p)
                db.set_config('receiver_email', recv_e)
                st.success("Security settings updated.")

        st.divider()
        
        c_gen1, c_gen2 = st.columns(2)
        mode = c_gen1.radio("System Mode", ["Corporate", "Education"])
        org = c_gen2.text_input("Organization Name", value=db.get_config('org_name'))
        
        if st.button("UPDATE SYSTEM"):
            db.set_config('system_mode', mode)
            db.set_config('org_name', org)
            st.success("System updated. Refresh to apply labels.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 8. MAIN NAVIGATION ROUTER
# ==========================================
def main():
    inject_titan_ui()
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️ SENTINEL AI")
        st.caption("TITAN SECURITY EDITION")
        st.divider()
        
        if st.session_state.page == 'kiosk':
            st.info("🟢 SYSTEM ARMED")
            
            # Camera Selection
            cam_opts = {0: "Webcam (Index 0)", 1: "External (Index 1)", 2: "IR/Depth (Index 2)"}
            sel_cam = st.selectbox("Video Input", list(cam_opts.keys()), format_func=lambda x: cam_opts[x], index=st.session_state.cam_index)
            if sel_cam != st.session_state.cam_index:
                st.session_state.cam_index = sel_cam
                st.session_state.camera_active = False # Restart
                st.rerun()
            
            st.divider()
            with st.expander("🔐 Admin Access"):
                p = st.text_input("Password", type="password")
                if st.button("Login"):
                    if p == db.get_config('admin_pwd'):
                        st.session_state.admin_auth = True
                        st.session_state.page = 'admin'
                        st.rerun()
                    else: st.error("Access Denied")
        
        elif st.session_state.page == 'admin':
            st.success("⚠️ ADMIN SESSION")
            if st.button("Exit to Kiosk"):
                st.session_state.page = 'kiosk'
                st.rerun()
        
        st.divider()
        if st.button("🌗 Switch Theme"):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()

    # Routing
    if st.session_state.page == 'kiosk':
        render_kiosk()
    elif st.session_state.page == 'admin':
        if st.session_state.admin_auth:
            render_admin()
        else:
            st.session_state.page = 'kiosk'
            st.rerun()

if __name__ == "__main__":
    main()