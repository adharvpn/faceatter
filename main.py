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
from datetime import datetime, timedelta, date
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 0. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sentinel AI | Ultra",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------
# GLOBAL CONSTANTS & PATHS
# ------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sentinel_ultra.db")
FACES_DATA_DIR = os.path.join(BASE_DIR, "biometric_vault")
UNKNOWN_LOGS_DIR = os.path.join(BASE_DIR, "security_breaches")
TRAINER_PATH = os.path.join(BASE_DIR, "neural_model.yml")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Ensure all critical directories exist
for folder in [FACES_DATA_DIR, UNKNOWN_LOGS_DIR, ASSETS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Audio Engine Initialization
try:
    import pygame
    pygame.mixer.init()
except ImportError:
    pass  # Fail silently if audio driver missing

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

# ==========================================
# 1. ADVANCED UI ENGINE (THE GLASS PROTOCOL)
# ==========================================
def inject_glass_ui():
    """
    Injects high-end CSS to override Streamlit defaults.
    Creates a translucent, frosted-glass look with neon accents.
    """
    
    # 1. Define Color Palettes based on Theme
    if st.session_state.theme == 'dark':
        # Dark Cyberpunk Palette
        bg_image = "https://images.unsplash.com/photo-1518544806314-5f855e86be25?q=80&w=2070&auto=format&fit=crop" # Dark abstract
        glass_bg = "rgba(18, 18, 25, 0.75)"
        glass_border = "1px solid rgba(255, 255, 255, 0.12)"
        text_color = "#e2e8f0"
        accent_color = "#00f2fe" # Neon Cyan
        accent_gradient = "linear-gradient(90deg, #00f2fe 0%, #4facfe 100%)"
        sidebar_bg = "rgba(10, 10, 15, 0.85)"
        shadow_light = "0 8px 32px 0 rgba(0, 0, 0, 0.5)"
    else:
        # Light Corporate Palette
        bg_image = "https://images.unsplash.com/photo-1497366216548-37526070297c?q=80&w=2069&auto=format&fit=crop" # Bright office
        glass_bg = "rgba(255, 255, 255, 0.65)"
        glass_border = "1px solid rgba(255, 255, 255, 0.6)"
        text_color = "#1e293b"
        accent_color = "#2563eb" # Royal Blue
        accent_gradient = "linear-gradient(90deg, #2563eb 0%, #4f46e5 100%)"
        sidebar_bg = "rgba(255, 255, 255, 0.85)"
        shadow_light = "0 8px 32px 0 rgba(31, 38, 135, 0.1)"

    # 2. Inject CSS
    st.markdown(f"""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

        /* RESET & BASE STYLES */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: {text_color};
        }}

        /* APP BACKGROUND */
        .stApp {{
            background-image: url('{bg_image}');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        
        /* SIDEBAR GLASS */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-right: {glass_border};
            box-shadow: 10px 0 20px rgba(0,0,0,0.2);
        }}

        /* GLASS PANELS (The Core Component) */
        .glass-card {{
            background: {glass_bg};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: {glass_border};
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: {shadow_light};
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            border-color: {accent_color};
        }}

        /* METRIC BOXES */
        .metric-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 15px;
            border-radius: 15px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .metric-label {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.8;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            background: {accent_gradient};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        /* BUTTON STYLING */
        .stButton>button {{
            background: {accent_gradient};
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            transition: all 0.2s ease;
            width: 100%;
            text-transform: uppercase;
            font-size: 0.9rem;
        }}
        .stButton>button:hover {{
            box-shadow: 0 0 20px {accent_color};
            transform: scale(1.02);
        }}

        /* DATAFRAME & TABLE STYLING */
        [data-testid="stDataFrame"] {{
            background: transparent !important;
        }}
        div[class*="stDataFrame"] div[class*="css"] {{
            background: transparent !important;
        }}

        /* INPUT FIELDS */
        .stTextInput>div>div, .stSelectbox>div>div, .stNumberInput>div>div {{
            background-color: rgba(255, 255, 255, 0.05);
            color: {text_color};
            border-radius: 10px;
            border: {glass_border};
        }}
        .stTextInput>div>div:focus-within {{
            border-color: {accent_color};
            box-shadow: 0 0 10px {accent_color}40;
        }}

        /* HEADERS */
        h1, h2, h3 {{
            font-weight: 800;
            letter-spacing: -1px;
        }}
        h1 span, h2 span {{
            color: {accent_color};
        }}

        /* CUSTOM SCROLLBAR */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(0,0,0,0.1); 
        }}
        ::-webkit-scrollbar-thumb {{
            background: {accent_color}; 
            border-radius: 4px;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATABASE ENGINE
# ==========================================
class DatabaseCore:
    def __init__(self):
        # check_same_thread=False is crucial for Streamlit's threading model
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._initialize_structure()

    def _initialize_structure(self):
        """Creates table schema if not exists. Supports Schema Migration logic."""
        try:
            # 1. Users (Employees/Students)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    dept TEXT,
                    role TEXT,
                    email TEXT,
                    phone TEXT,
                    face_id INTEGER UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. Attendance Logs
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    date TEXT,
                    time TEXT,
                    type TEXT,   -- Check-In, Check-Out, Manual, Auto-Absent
                    status TEXT, -- Present, Late, Absent, Leave, WFH, OD
                    confidence REAL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            # 3. Holiday Calendar
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS holidays (
                    date TEXT PRIMARY KEY,
                    description TEXT,
                    type TEXT -- Public, Optional, Emergency
                )
            """)
            
            # 4. System Settings (Key-Value Store)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY, 
                    value TEXT
                )
            """)
            
            # 5. Populate Default Settings if empty
            default_config = {
                'admin_pwd': 'admin',
                'system_mode': 'Corporate',  # or 'Education'
                'shift_start': '09:00',
                'shift_end': '17:00',
                'attendance_threshold': '75', # For students
                'camera_id': '0',
                'org_name': 'Sentinel Corp'
            }
            
            for k, v in default_config.items():
                self.cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
            
            self.conn.commit()
            
        except Exception as e:
            st.error(f"Database Initialization Failed: {e}")

    # --- SETTINGS MANAGER ---
    def get_config(self, key):
        self.cursor.execute("SELECT value FROM settings WHERE key=?", (key,))
        res = self.cursor.fetchone()
        return res[0] if res else None

    def set_config(self, key, value):
        self.cursor.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        self.conn.commit()

    # --- ATTENDANCE MANAGER ---
    def log_punch(self, user_id, type_, status, conf=100.0):
        now = datetime.now()
        d_str = now.strftime("%Y-%m-%d")
        t_str = now.strftime("%H:%M:%S")
        
        try:
            self.cursor.execute("""
                INSERT INTO attendance (user_id, date, time, type, status, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, d_str, t_str, type_, status, conf))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Log Error: {e}")
            return False

    def manual_override(self, user_id, date_str, status):
        """Allows Admin to manually fix attendance."""
        try:
            # First, clear any existing auto-logs for that day to avoid conflict
            # Or insert a specific 'Manual' record that takes precedence in reporting
            self.cursor.execute("""
                INSERT INTO attendance (user_id, date, time, type, status, confidence)
                VALUES (?, ?, '09:00:00', 'Manual', ?, 100)
            """, (user_id, date_str, status))
            self.conn.commit()
            return True
        except: return False

    # --- HOLIDAY MANAGER ---
    def add_holiday(self, date_obj, desc):
        d_str = date_obj.strftime("%Y-%m-%d")
        try:
            self.cursor.execute("INSERT INTO holidays (date, description, type) VALUES (?, ?, 'Public')", (d_str, desc))
            self.conn.commit()
            return True
        except: return False

# Initialize DB globally
db = DatabaseCore()

# ==========================================
# 3. BIOMETRIC INTELLIGENCE LAYER
# ==========================================
class FaceNet:
    def __init__(self):
        # Using Haar Cascades for speed on CPU
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # LBPH (Local Binary Patterns Histograms) for recognition
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_loaded = False
        self.load_weights()

    def load_weights(self):
        if os.path.exists(TRAINER_PATH):
            try:
                self.recognizer.read(TRAINER_PATH)
                self.model_loaded = True
            except Exception as e:
                print(f"Model Load Error: {e}")
                self.model_loaded = False

    def train(self):
        """Iterates through data folders and retrains the model."""
        faces, ids = [], []
        
        if not os.path.exists(FACES_DATA_DIR):
            return False, "No Data Directory"

        try:
            user_folders = os.listdir(FACES_DATA_DIR)
            for folder in user_folders:
                path = os.path.join(FACES_DATA_DIR, folder)
                if not os.path.isdir(path): continue
                
                # Folder name must be the Integer Face ID
                try:
                    face_id = int(folder)
                except ValueError: continue # Skip non-integer folders

                for img_name in os.listdir(path):
                    if img_name.startswith("."): continue # Skip system files
                    img_path = os.path.join(path, img_name)
                    
                    # Read and convert to grayscale
                    img_pil = Image.open(img_path).convert('L') 
                    img_numpy = np.array(img_pil, 'uint8')
                    
                    faces.append(img_numpy)
                    ids.append(face_id)
            
            if len(faces) > 0:
                self.recognizer.train(faces, np.array(ids))
                self.recognizer.save(TRAINER_PATH)
                self.model_loaded = True
                return True, f"Trained on {len(faces)} samples."
            else:
                return False, "No face data found."
                
        except Exception as e:
            return False, str(e)

# Initialize AI globally
ai_engine = FaceNet()

# ==========================================
# 4. UTILITY FUNCTIONS
# ==========================================
def play_audio_feedback(type="success"):
    """Plays sound effects for user feedback."""
    try:
        if type == "success":
            # You can place a 'success.wav' in assets folder
            sound_file = os.path.join(ASSETS_DIR, "success.wav")
        else:
            sound_file = os.path.join(ASSETS_DIR, "alert.wav")
            
        if os.path.exists(sound_file):
            pygame.mixer.Sound(sound_file).play()
    except: pass

def get_dynamic_labels():
    """Returns terminology based on Corporate vs Education mode."""
    mode = db.get_config('system_mode')
    if mode == 'Education':
        return {
            'user': 'Student',
            'users': 'Students',
            'dept': 'Class/Section',
            'id': 'Roll Number',
            'role': 'Stream',
            'threshold': True
        }
    else:
        return {
            'user': 'Employee',
            'users': 'Employees',
            'dept': 'Department',
            'id': 'Employee ID',
            'role': 'Designation',
            'threshold': False
        }

# ==========================================
# 5. PAGE: SMART KIOSK (Main Interface)
# ==========================================
def render_kiosk():
    # Load settings
    labels = get_dynamic_labels()
    org_name = db.get_config('org_name')
    shift_start = db.get_config('shift_start')
    
    # Header Section
    c_head, c_time = st.columns([3, 1])
    with c_head:
        st.markdown(f"<h1 style='font-size: 3rem;'>{org_name} <span style='font-size:1.5rem; opacity:0.6'>| Smart Access</span></h1>", unsafe_allow_html=True)
    with c_time:
        st.markdown(f"""
        <div style='text-align:right; font-family: "JetBrains Mono";'>
            <div style='font-size: 2rem; color: #00f2fe; font-weight:bold;'>{datetime.now().strftime('%H:%M')}</div>
            <div style='font-size: 0.9rem; opacity: 0.7;'>{datetime.now().strftime('%A, %d %B')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Main Grid
    col_video, col_stats = st.columns([1.8, 1])

    with col_video:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        # We use a placeholder to update the image without full rerun
        camera_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        # Control Bar
        c_btn1, c_btn2 = st.columns(2)
        start_cam = c_btn1.button("🟢 INITIALIZE SCANNER", use_container_width=True)
        stop_cam = c_btn2.button("🔴 TERMINATE FEED", use_container_width=True)

        if start_cam: st.session_state.camera_active = True
        if stop_cam: st.session_state.camera_active = False

    with col_stats:
        # Live Stats Panel
        today = datetime.now().strftime("%Y-%m-%d")
        
        # SQL Queries for stats
        present_count = db.cursor.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date=?", (today,)).fetchone()[0]
        
        st.markdown(f"""
        <div class='glass-card'>
            <h3 style='margin-top:0'>📊 Live Pulse</h3>
            <div class='metric-container'>
                <div class='metric-label'>CHECKED IN TODAY</div>
                <div class='metric-value'>{present_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='glass-card'><h3>📝 Recent Logs</h3>", unsafe_allow_html=True)
        
        # Fetch last 5 logs joined with user names
        logs = db.cursor.execute(f"""
            SELECT u.name, a.time, a.status 
            FROM attendance a 
            JOIN users u ON a.user_id = u.id 
            WHERE a.date='{today}' 
            ORDER BY a.log_id DESC LIMIT 5
        """).fetchall()
        
        if logs:
            df_log = pd.DataFrame(logs, columns=["Name", "Time", "Status"])
            st.dataframe(df_log, hide_index=True, use_container_width=True)
        else:
            st.info("No activity recorded yet.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # --- CAMERA LOGIC LOOP ---
    if st.session_state.camera_active:
        # Get mapping of FaceID -> User Data for quick lookup
        all_users = db.cursor.execute("SELECT face_id, name, id FROM users").fetchall()
        user_map = {r[0]: {'name': r[1], 'uid': r[2]} for r in all_users}
        
        cap = cv2.VideoCapture(int(st.session_state.cam_index))
        
        while st.session_state.camera_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera Input Failed.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = ai_engine.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                # Default: Unknown
                name = "UNKNOWN"
                color = (0, 0, 255) # Red
                
                try:
                    if ai_engine.model_loaded:
                        id_, confidence = ai_engine.recognizer.predict(gray[y:y+h, x:x+w])
                        
                        # Confidence: Lower is better in LBPH (0 = perfect match)
                        if confidence < 65 and id_ in user_map:
                            u_data = user_map[id_]
                            name = u_data['name']
                            uid = u_data['uid']
                            color = (0, 255, 0) # Green
                            
                            # --- ATTENDANCE LOGIC ---
                            # 1. Cooldown Check (Prevent spamming DB)
                            now = datetime.now()
                            last_time = st.session_state.last_log_time.get(uid, datetime.min)
                            
                            if (now - last_time).total_seconds() > 300: # 5 Minutes cooldown
                                # 2. Determine Check-In vs Check-Out
                                count = db.cursor.execute("SELECT COUNT(*) FROM attendance WHERE user_id=? AND date=?", (uid, today)).fetchone()[0]
                                type_ = "Check-In" if count == 0 else "Check-Out"
                                
                                # 3. Status Logic (Late/OnTime)
                                status = "Present"
                                if type_ == "Check-In":
                                    # Simple string comparison works for HH:MM format
                                    if now.strftime("%H:%M") > shift_start:
                                        status = "Late"
                                    else:
                                        status = "On Time"
                                
                                # 4. Log to DB
                                if db.log_punch(uid, type_, status):
                                    st.session_state.last_log_time[uid] = now
                                    play_audio_feedback("success")
                                    st.toast(f"✅ {name} Marked {status} ({type_})")
                                    
                except Exception as e:
                    print(e)

                # Draw UI on Frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                # Futuristic corners
                cv2.line(frame, (x,y), (x+20, y), color, 4)
                cv2.line(frame, (x,y), (x, y+20), color, 4)
                
                # Name Badge
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                cv2.putText(frame, name, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame, channels="RGB", use_container_width=True)
            
            # Simple sleep to reduce CPU usage
            time.sleep(0.01)
        
        cap.release()
        camera_placeholder.empty()


# ==========================================
# 6. PAGE: ADMIN COMMAND CENTER
# ==========================================
def render_admin():
    labels = get_dynamic_labels()
    
    # Logout Header
    c1, c2 = st.columns([8, 1])
    c1.title("COMMAND CENTER")
    if c2.button("LOGOUT"):
        st.session_state.admin_auth = False
        st.session_state.page = 'kiosk'
        st.rerun()

    # Admin Tabs
    tab_dash, tab_users, tab_report, tab_settings = st.tabs([
        "📊 Analytics & Insights", 
        f"👥 {labels['users']} Management", 
        "📅 Reports & Payroll", 
        "⚙️ System Config"
    ])

    # --- TAB 1: ANALYTICS ---
    with tab_dash:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Date Filter
        filter_date = st.date_input("Select Date", date.today())
        d_str = filter_date.strftime("%Y-%m-%d")

        # Fetch Data
        df = pd.read_sql_query(f"""
            SELECT u.dept, a.status, a.time, a.type 
            FROM attendance a 
            JOIN users u ON a.user_id = u.id 
            WHERE a.date='{d_str}'
        """, db.conn)

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        total_staff = db.cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        present = df['status'].count()
        absent = total_staff - df[df['type']=='Check-In'].shape[0] # Roughly
        
        col_kpi1.markdown(f"<div class='glass-card'><div class='metric-label'>TOTAL {labels['users'].upper()}</div><div class='metric-value'>{total_staff}</div></div>", unsafe_allow_html=True)
        col_kpi2.markdown(f"<div class='glass-card'><div class='metric-label'>PRESENT</div><div class='metric-value'>{df[df['type']=='Check-In'].shape[0]}</div></div>", unsafe_allow_html=True)
        col_kpi3.markdown(f"<div class='glass-card'><div class='metric-label'>LATE ARRIVALS</div><div class='metric-value'>{df[df['status']=='Late'].shape[0]}</div></div>", unsafe_allow_html=True)

        # Charts
        c_chart1, c_chart2 = st.columns(2)
        
        with c_chart1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Arrival Timeline")
            if not df.empty:
                df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
                fig = px.histogram(df[df['type']=='Check-In'], x='hour', nbins=12, template="plotly_dark", color_discrete_sequence=['#00f2fe'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No data available")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_chart2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader(f"{labels['dept']} Distribution")
            if not df.empty:
                fig2 = px.pie(df, names='dept', hole=0.6, template="plotly_dark")
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.caption("No data available")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: USER MANAGEMENT ---
    with tab_users:
        st.markdown("<br>", unsafe_allow_html=True)
        
        c_add, c_list = st.columns([1, 1.5])
        
        with c_add:
            st.markdown(f"<div class='glass-card'><h3>➕ Onboard {labels['user']}</h3>", unsafe_allow_html=True)
            with st.form("add_user_form"):
                u_id = st.text_input(labels['id'])
                u_name = st.text_input("Full Name")
                u_dept = st.selectbox(labels['dept'], ["Engineering", "Sales", "HR", "Operations", "Finance", "Class 10-A", "Class 10-B"])
                u_role = st.text_input(labels['role'])
                u_email = st.text_input("Email Address")
                
                if st.form_submit_button("CREATE PROFILE"):
                    try:
                        # Auto-assign Face ID
                        curr_max = db.cursor.execute("SELECT MAX(face_id) FROM users").fetchone()[0]
                        new_face_id = 1 if curr_max is None else curr_max + 1
                        
                        db.cursor.execute("INSERT INTO users (id, name, dept, role, email, face_id) VALUES (?,?,?,?,?,?)", 
                                         (u_id, u_name, u_dept, u_role, u_email, new_face_id))
                        db.conn.commit()
                        st.success(f"User Created! Assigned System ID: {new_face_id}")
                    except sqlite3.IntegrityError:
                        st.error(f"{labels['id']} already exists!")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # --- BIOMETRIC TRAINING WIDGET ---
            st.markdown(f"<div class='glass-card'><h3>🧬 Biometric Enrollment</h3>", unsafe_allow_html=True)
            
            # Select User to Train
            users_list = db.cursor.execute("SELECT face_id, name, id FROM users").fetchall()
            user_opts = {u[0]: f"{u[1]} ({u[2]})" for u in users_list}
            
            sel_face_id = st.selectbox("Select Profile to Train", options=list(user_opts.keys()), format_func=lambda x: user_opts[x])
            
            if st.button("📸 START CAPTURE SEQUENCE"):
                save_path = os.path.join(FACES_DATA_DIR, str(sel_face_id))
                os.makedirs(save_path, exist_ok=True)
                
                cap = cv2.VideoCapture(int(st.session_state.cam_index))
                progress_bar = st.progress(0)
                status_txt = st.empty()
                
                count = 0
                while count < 50:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = ai_engine.detector.detectMultiScale(gray, 1.2, 5)
                    
                    for (x,y,w,h) in faces:
                        count += 1
                        cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), gray[y:y+h, x:x+w])
                        # Visual feedback on frame not needed here as we are capturing fast
                        
                    progress_bar.progress(count / 50)
                    status_txt.text(f"Capturing samples: {count}/50")
                    time.sleep(0.05)
                
                cap.release()
                
                # Trigger Retrain
                with st.spinner("Compiling Neural Model..."):
                    success, msg = ai_engine.train()
                    if success: st.success("Biometric Profile Active!")
                    else: st.error(f"Training Failed: {msg}")

            st.markdown("</div>", unsafe_allow_html=True)

        with c_list:
            st.markdown(f"<div class='glass-card'><h3>📂 {labels['users']} Directory</h3>", unsafe_allow_html=True)
            df_users = pd.read_sql_query("SELECT id, name, dept, role, email, face_id FROM users", db.conn)
            st.dataframe(df_users, use_container_width=True, height=600)
            
            # Delete Action
            del_uid = st.text_input(f"Remove {labels['user']} by ID")
            if st.button("DELETE RECORD"):
                if del_uid:
                    db.cursor.execute("DELETE FROM users WHERE id=?", (del_uid,))
                    db.conn.commit()
                    st.warning("User deleted. Please retrain model to remove artifacts.")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 3: REPORTING & MANUAL FIXES ---
    with tab_report:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1. Advanced Matrix Report
        st.markdown("<div class='glass-card'><h3>🗓️ Monthly Attendance Matrix</h3>", unsafe_allow_html=True)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        r_month = col_r1.selectbox("Month", range(1, 13), index=datetime.now().month-1)
        r_year = col_r2.number_input("Year", value=datetime.now().year)
        
        if col_r3.button("GENERATE MATRIX"):
            # Logic to generate P/A/L grid
            num_days = calendar.monthrange(r_year, r_month)[1]
            all_u = db.cursor.execute("SELECT id, name FROM users").fetchall()
            
            matrix_data = []
            
            for uid, uname in all_u:
                row = {"ID": uid, "Name": uname}
                stats = {'P': 0, 'A': 0, 'L': 0}
                
                for day in range(1, num_days + 1):
                    d_date = date(r_year, r_month, day).strftime("%Y-%m-%d")
                    
                    # Fetch logs
                    # Check for Holiday
                    hol = db.cursor.execute("SELECT description FROM holidays WHERE date=?", (d_date,)).fetchone()
                    
                    # Check Attendance
                    att = db.cursor.execute("SELECT status FROM attendance WHERE user_id=? AND date=?", (uid, d_date)).fetchone()
                    
                    code = "A" # Default Absent
                    if hol: code = "H"
                    elif att:
                        s = att[0]
                        if s in ["Present", "On Time", "Late"]: 
                            code = "P"
                            stats['P'] += 1
                        elif s == "Leave": code = "L"
                        elif s == "Work From Home": code = "WFH"
                    
                    row[str(day)] = code
                
                # Calculate Score
                total_working = num_days # Simplified
                score = (stats['P'] / total_working) * 100
                row['Score'] = f"{score:.1f}%"
                matrix_data.append(row)
            
            df_matrix = pd.DataFrame(matrix_data)
            
            # Color Styling
            def color_matrix(val):
                color = ''
                if val == 'P': color = 'color: #00ff00'
                elif val == 'A': color = 'color: #ff4b4b'
                elif val == 'H': color = 'color: #ffa500'
                elif str(val).endswith('%'):
                    score = float(val.strip('%'))
                    if score < 75: color = 'color: red; font-weight: bold'
                return color

            st.dataframe(df_matrix.style.applymap(color_matrix), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 2. Manual Corrections
        st.markdown("<div class='glass-card'><h3>🛠️ Attendance Correction Tool</h3>", unsafe_allow_html=True)
        c_fix1, c_fix2 = st.columns(2)
        
        fix_uid = c_fix1.text_input(f"Enter {labels['id']} to Fix")
        fix_date = c_fix2.date_input("Date of correction")
        fix_status = st.selectbox("Set Status To", ["Present", "Work From Home", "Sick Leave", "Official Duty", "Absent"])
        
        if st.button("UPDATE RECORD"):
            db.manual_override(fix_uid, fix_date.strftime("%Y-%m-%d"), fix_status)
            st.success("Record Updated Successfully.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 4: SETTINGS ---
    with tab_settings:
        st.markdown("<div class='glass-card'><h3>⚙️ Core Configuration</h3>", unsafe_allow_html=True)
        
        # Mode Switch
        curr_mode = db.get_config('system_mode')
        new_mode = st.radio("Operating Mode", ["Corporate", "Education"], index=0 if curr_mode=='Corporate' else 1, horizontal=True)
        if new_mode != curr_mode:
            db.set_config('system_mode', new_mode)
            st.rerun()
            
        st.divider()
        
        # Org Details
        new_name = st.text_input("Organization Name", value=db.get_config('org_name'))
        if st.button("Save Name"):
            db.set_config('org_name', new_name)
            st.success("Saved")
            
        c_s1, c_s2 = st.columns(2)
        t_start = c_s1.text_input("Shift Start", db.get_config('shift_start'))
        t_end = c_s2.text_input("Shift End", db.get_config('shift_end'))
        if st.button("Update Timings"):
            db.set_config('shift_start', t_start)
            db.set_config('shift_end', t_end)
            st.success("Timings updated")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Holiday Manager
        st.markdown("<div class='glass-card'><h3>🎉 Holiday Calendar</h3>", unsafe_allow_html=True)
        c_h1, c_h2 = st.columns([1, 2])
        h_date = c_h1.date_input("Holiday Date")
        h_desc = c_h2.text_input("Occasion")
        if st.button("Add Holiday"):
            if db.add_holiday(h_date, h_desc): st.success("Added")
            else: st.error("Failed")
            
        # List Holidays
        hols = pd.read_sql_query("SELECT * FROM holidays", db.conn)
        st.dataframe(hols, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 7. AUTHENTICATION & ROUTING
# ==========================================
def render_sidebar_nav():
    with st.sidebar:
        st.markdown("## 🛡️ NAVIGATOR")
        
        if st.session_state.page == 'kiosk':
            st.info("Status: MONITORING ACTIVE")
            
            # Camera Switcher
            cam_options = {0: "Webcam (Index 0)", 1: "External Cam (Index 1)", 2: "IR Cam (Index 2)"}
            sel = st.selectbox("Video Source", list(cam_options.keys()), format_func=lambda x: cam_options[x], index=st.session_state.cam_index)
            if sel != st.session_state.cam_index:
                st.session_state.cam_index = sel
                st.session_state.camera_active = False # Reset camera
                st.rerun()
                
            st.divider()
            
            # Admin Login Expander
            with st.expander("🔐 Admin Access"):
                pwd = st.text_input("Password", type="password")
                if st.button("Login"):
                    if pwd == db.get_config('admin_pwd'):
                        st.session_state.admin_auth = True
                        st.session_state.page = 'admin'
                        st.rerun()
                    else:
                        st.error("Access Denied")
        
        elif st.session_state.page == 'admin':
            st.success("Status: ADMIN SESSION")
            if st.button("Return to Kiosk"):
                st.session_state.admin_auth = False
                st.session_state.page = 'kiosk'
                st.rerun()
                
        st.divider()
        # Theme Toggle
        if st.button("🌗 Toggle Theme"):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()

# ==========================================
# 8. MAIN ENTRY POINT
# ==========================================
def main():
    inject_glass_ui()
    render_sidebar_nav()
    
    if st.session_state.page == 'kiosk':
        render_kiosk()
    elif st.session_state.page == 'admin':
        if st.session_state.admin_auth:
            render_admin()
        else:
            st.error("Unauthorized. Please log in.")
            st.session_state.page = 'kiosk'
            st.rerun()

if __name__ == "__main__":
    main()