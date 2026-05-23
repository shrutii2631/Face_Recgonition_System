import os, cv2, numpy as np, pandas as pd
from prometheus_flask_exporter import PrometheusMetrics
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.utils import secure_filename

# ========= CONFIG =========
USERNAME = "admin"
PASSWORD = "admin"

DATASET_PATH = "dataset"
MODEL_DIR = "models"
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "attendance.csv")
REG_MAP_FILE = os.path.join(DATA_DIR, "reg_map.csv")
REGISTER_IMAGES_COUNT = 25

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ========= FACE SETUP =========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
MODEL_FILE = os.path.join(MODEL_DIR, "trainer.yml")

label_map = {}           # label_id -> reg_no
recognized_today = {}    # reg_no -> name
last_date = None

# ========= CSV INIT =========
if not os.path.exists(REG_MAP_FILE):
    pd.DataFrame(columns=["Reg_No", "Name"]).to_csv(REG_MAP_FILE, index=False)

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Reg_No", "Name", "Date", "In_Time", "Out_Time", "Attendance"]).to_csv(CSV_FILE, index=False)

reg_df = pd.read_csv(REG_MAP_FILE, dtype=str)
df = pd.read_csv(CSV_FILE, dtype=str)

# ========= HELPERS =========
def sync_reg_map_with_dataset():
    global reg_df
    dataset_regs = {d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))}
    csv_regs = set(reg_df["Reg_No"].astype(str))

    for reg_no in dataset_regs - csv_regs:
        reg_df.loc[len(reg_df)] = [reg_no, f"User_{reg_no}"]

    reg_df.to_csv(REG_MAP_FILE, index=False)

def load_label_map():
    label_map.clear()
    label = 0
    for reg_no in sorted(os.listdir(DATASET_PATH)):
        if os.path.isdir(os.path.join(DATASET_PATH, reg_no)):
            label_map[label] = reg_no
            label += 1

def retrain_model():
    faces, labels = [], []
    label = 0
    for reg_no in sorted(os.listdir(DATASET_PATH)):
        path = os.path.join(DATASET_PATH, reg_no)
        if not os.path.isdir(path):
            continue
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_gray = cv2.imread(img_path, 0)
            if img_gray is not None:
                faces.append(img_gray)
                labels.append(label)
        label += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_FILE)

def load_model_if_exists():
    if os.path.exists(MODEL_FILE):
        recognizer.read(MODEL_FILE)

sync_reg_map_with_dataset()
load_label_map()
load_model_if_exists()
if os.listdir(DATASET_PATH) and not os.path.exists(MODEL_FILE):
    retrain_model()

# ========= FLASK APP =========
app = Flask(__name__)
metrics = PrometheusMetrics(app)
app.secret_key = "super-secret-key-change-me"

# ========= LOGIN =========
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class Admin(UserMixin):
    id = "admin"

@login_manager.user_loader
def load_user(user_id):
    if user_id == "admin":
        return Admin()
    return None

@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("dashboard"))

@app.route("/health")
def health():
    return {"status": "healthy"}, 200

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")
        if u == USERNAME and p == PASSWORD:
            login_user(Admin())
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username/password")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ========= PAGES =========
@app.route("/dashboard")
@login_required
def dashboard():
    global df
    df = pd.read_csv(CSV_FILE, dtype=str)
    total_students = len(reg_df)
    today = datetime.now().date().isoformat()
    today_present = len(df[df["Date"] == today]) if "Date" in df.columns else 0
    return render_template("dashboard.html",
                           total_students=total_students,
                           today=today,
                           today_present=today_present)

@app.route("/register")
@login_required
def register_page():
    return render_template("register.html", count=REGISTER_IMAGES_COUNT)

@app.route("/attendance")
@login_required
def attendance_page():
    global df
    df = pd.read_csv(CSV_FILE, dtype=str)
    rows = df.sort_values(by=["Date","In_Time"], ascending=False).head(200).to_dict(orient="records")
    return render_template("attendance.html", rows=rows)

# ========= API: REGISTER (receive images) =========
@app.route("/api/register/start", methods=["POST"])
@login_required
def api_register_start():
    reg_no = request.json.get("reg_no", "").strip()
    name = request.json.get("name", "").strip()

    if not reg_no or not name:
        return jsonify({"ok": False, "msg": "Name & Reg No required"}), 400

    global reg_df
    reg_df = pd.read_csv(REG_MAP_FILE, dtype=str)

    if reg_no in reg_df["Reg_No"].values:
        return jsonify({"ok": False, "msg": "Registration already exists"}), 400

    save_path = os.path.join(DATASET_PATH, reg_no)
    os.makedirs(save_path, exist_ok=True)
    return jsonify({"ok": True, "msg": "Started", "save_path": save_path})

@app.route("/api/register/frame", methods=["POST"])
@login_required
def api_register_frame():
    """
    Frontend sends a JPG frame (multipart/form-data) + reg_no + idx
    We detect exactly ONE face, crop to 200x200 grayscale, save as idx.jpg
    """
    reg_no = request.form.get("reg_no", "").strip()
    idx = request.form.get("idx", "").strip()
    if "frame" not in request.files or not reg_no or idx == "":
        return jsonify({"ok": False, "msg": "Invalid payload"}), 400

    file = request.files["frame"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "msg": "Bad image"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 1:
        return jsonify({"ok": False, "msg": "Need exactly ONE face"}), 200

    x, y, w, h = faces[0]
    face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

    save_path = os.path.join(DATASET_PATH, reg_no)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, f"{idx}.jpg"), face_img)

    return jsonify({"ok": True, "msg": "Saved"})

@app.route("/api/register/finish", methods=["POST"])
@login_required
def api_register_finish():
    reg_no = request.json.get("reg_no", "").strip()
    name = request.json.get("name", "").strip()

    if not reg_no or not name:
        return jsonify({"ok": False, "msg": "Missing data"}), 400

    global reg_df
    reg_df = pd.read_csv(REG_MAP_FILE, dtype=str)
    reg_df.loc[len(reg_df)] = [reg_no, name]
    reg_df.to_csv(REG_MAP_FILE, index=False)

    load_label_map()
    retrain_model()

    return jsonify({"ok": True, "msg": "Student registered & model trained"})

# ========= API: ATTENDANCE RECOGNITION =========
@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
    print("✅ /api/recognize HIT", flush=True)
    global df, last_date, recognized_today, reg_df

    if "frame" not in request.files:
        return jsonify({"ok": False, "msg": "Missing frame"}), 400

    # reset day cache
    today = datetime.now().date().isoformat()
    if last_date != today:
        recognized_today.clear()
        last_date = today

    file = request.files["frame"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "msg": "Bad image"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    reg_df = pd.read_csv(REG_MAP_FILE, dtype=str)
    df = pd.read_csv(CSV_FILE, dtype=str)

    now = datetime.now().strftime("%H:%M:%S")

    for (x,y,w,h) in faces[:3]:  # limit faces
        face_img = cv2.resize(gray[y:y+h, x:x+w], (200,200))

        try:
            label_id, confidence = recognizer.predict(face_img)
            print("PRED => label:", label_id, "conf:", confidence)
        except:
            continue

        accuracy = max(0, min(100, int(100 - confidence * 1.5)))

        name = "Unknown"
        reg_no = "Unknown"

        if label_id in label_map and confidence < 90:
            reg_no = label_map[label_id]
            row = reg_df.loc[reg_df["Reg_No"] == reg_no]
            if not row.empty:
                name = row["Name"].values[0]
                recognized_today[reg_no] = name

                # mark attendance
                if not ((df["Reg_No"] == reg_no) & (df["Date"] == today)).any():
                    df.loc[len(df)] = [reg_no, name, today, now, now, "Present"]
                else:
                    df.loc[(df["Reg_No"] == reg_no) & (df["Date"] == today), "Out_Time"] = now

        results.append({
            "box": [int(x), int(y), int(w), int(h)],
            "name": name,
            "reg_no": reg_no,
            "accuracy": accuracy
        })

    df.to_csv(CSV_FILE, index=False)
    return jsonify({"ok": True, "faces": results})

# ========= EXPORT =========
@app.route("/export/today")
@login_required
def export_today():
    df = pd.read_csv(CSV_FILE, dtype=str)
    today = datetime.now().date().isoformat()
    report = df[df["Date"] == today]
    if report.empty:
        return "No data today", 400
    file = os.path.join(DATA_DIR, f"attendance_{today}.xlsx")
    report.to_excel(file, index=False)
    return send_file(file, as_attachment=True)

@app.route("/export/range", methods=["POST"])
@login_required
def export_range():
    df = pd.read_csv(CSV_FILE, dtype=str)
    from_date = request.form.get("from_date")
    to_date = request.form.get("to_date")

    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
        to_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
    except:
        return "Invalid date format", 400

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    report = df[(df["Date"] >= from_dt) & (df["Date"] <= to_dt)]

    if report.empty:
        return "No attendance in this range", 400

    file = os.path.join(DATA_DIR, f"attendance_{from_date}_to_{to_date}.xlsx")
    report.to_excel(file, index=False)
    return send_file(file, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)