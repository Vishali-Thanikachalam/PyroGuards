import cv2
import numpy as np
import os
import time
import webbrowser
from threading import Thread, Lock
from flask import Flask, request, jsonify, Response, render_template_string
from inference import get_model
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
ROBOFLOW_API_KEY = "Es83n4ozIQtovETsWtVP"
MODEL_ID = "fire-aonru-pxb8c/1"
CONFIG_FILE = "zone_configs.npy"

CAMERA_INDEX = 1
FRAME_W, FRAME_H = 1280, 720
CONF_THRESHOLD = 0.5
PERSISTENCE_THRESHOLD = 2.0 
TREND_WINDOW = 15
GAS_THRESHOLD = 3000

# =========================================================
# SHARED STATE
# =========================================================
state_lock = Lock()
shared_state = {
    "status": "SAFE",
    "sensor_data": {'temp': 0, 'hum': 0, 'gas': 0, 'fire': 1},
    "temp_threshold": 32,
    "total_people": 0,
    "zone_info": {} 
}

aisle_configs = []
current_pts = []
latest_frame = None
main_view = None # Frame with drawings for Dashboard
latest_predictions = [] 
is_running = True
zone_timers = {}
zone_area_history = {}

# =========================================================
# AI MODELS
# =========================================================
os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY
fire_model = get_model(MODEL_ID)
person_model = YOLO("yolov8n.pt") 

# =========================================================
# DASHBOARD HTML TEMPLATE
# =========================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PYROGUARD V4 | Dashboard</title>
    <!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PYROGUARD V4 - AI Fire Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --accent-fire: #ff4444;
            --accent-safe: #00cc88;
            --accent-warning: #ffaa00;
            --glow-fire: 0 0 30px rgba(255, 68, 68, 0.6);
            --glow-safe: 0 0 20px rgba(0, 204, 136, 0.5);
            --glow-warning: 0 0 25px rgba(255, 170, 0, 0.6);
            --text-primary: #e8e8e8;
            --text-secondary: #b0b0b0;
            --glass-bg: rgba(255, 255, 255, 0.05);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body { 
            font-family: 'Roboto', sans-serif; 
            background: linear-gradient(135deg, var(--bg-primary) 0%, #1a0f1a 50%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            margin: 0; 
            display: flex; 
            height: 100vh; 
            overflow: hidden;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(255, 68, 68, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 40% 40%, rgba(0, 204, 136, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        .sidebar { 
            width: 380px; 
            background: linear-gradient(180deg, rgba(26, 26, 26, 0.95) 0%, rgba(10, 10, 10, 0.98) 100%);
            backdrop-filter: blur(20px);
            padding: 30px 25px;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            display: flex; 
            flex-direction: column;
            box-shadow: 5px 0 30px rgba(0, 0, 0, 0.5);
            z-index: 2;
            position: relative;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 35px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255, 68, 68, 0.3);
        }

        .logo i {
            font-size: 32px;
            background: linear-gradient(135deg, var(--accent-fire), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 10px rgba(255, 68, 68, 0.5));
            animation: logoPulse 2s ease-in-out infinite;
        }

        @keyframes logoPulse {
            0%, 100% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(255, 68, 68, 0.5)); }
            50% { transform: scale(1.05); filter: drop-shadow(0 0 20px rgba(255, 68, 68, 0.8)); }
        }

        .logo h2 {
            font-family: 'Orbitron', monospace;
            font-weight: 900;
            font-size: 24px;
            background: linear-gradient(135deg, var(--accent-fire), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
            margin: 0;
        }

        .status-box { 
            padding: 25px;
            border-radius: 16px;
            text-align: center; 
            font-size: 22px; 
            font-weight: 800; 
            margin-bottom: 30px; 
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            font-family: 'Orbitron', monospace;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .status-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        .status-box:hover::before {
            left: 100%;
        }

        .SAFE { 
            background: linear-gradient(135deg, rgba(27, 94, 32, 0.8) 0%, rgba(0, 204, 136, 0.6) 100%);
            color: #c8e6c9;
            border: 1px solid rgba(76, 175, 80, 0.5);
            box-shadow: var(--glow-safe);
        }

        .WARNING { 
            background: linear-gradient(135deg, rgba(230, 81, 0, 0.8) 0%, rgba(255, 170, 0, 0.6) 100%);
            color: #ffcc80;
            border: 1px solid rgba(255, 152, 0, 0.5);
            box-shadow: var(--glow-warning);
        }

        .FIRE { 
            background: linear-gradient(135deg, rgba(183, 28, 28, 0.9) 0%, rgba(255, 68, 68, 0.8) 100%);
            color: #ffcdd2;
            border: 1px solid rgba(244, 67, 54, 0.7);
            box-shadow: var(--glow-fire);
            animation: pulse 1s infinite, alarmShake 0.1s infinite;
            transform-origin: center;
        }

        @keyframes alarmShake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-2px); }
            75% { transform: translateX(2px); }
        }

        @keyframes pulse { 
            0% { box-shadow: var(--glow-fire); }
            50% { box-shadow: 0 0 50px rgba(255, 68, 68, 1); }
            100% { box-shadow: var(--glow-fire); }
        }

        .stat-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 18px; 
            margin-bottom: 30px;
        }

        .stat-card { 
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            padding: 20px 18px;
            border-radius: 12px; 
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-fire), var(--accent-safe), var(--accent-warning));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .stat-card:hover::before {
            opacity: 1;
        }

        .stat-card i {
            font-size: 24px;
            margin-bottom: 10px;
            opacity: 0.7;
        }

        .stat-label { 
            font-size: 11px; 
            color: var(--text-secondary);
            text-transform: uppercase; 
            margin-bottom: 8px; 
            letter-spacing: 1px;
            font-weight: 500;
        }

        .stat-value { 
            font-size: 24px; 
            font-weight: 700; 
            color: var(--text-primary);
            font-family: 'Orbitron', monospace;
            letter-spacing: 0.5px;
        }

        .main-view { 
            flex-grow: 1; 
            background: radial-gradient(ellipse at center, rgba(0,0,0,0.95) 0%, rgba(10,10,10,1) 70%);
            display: flex; 
            align-items: center; 
            justify-content: center; 
            position: relative;
            overflow: hidden;
        }

        .video-feed { 
            width: 98%; 
            height: 98%; 
            max-width: 1400px;
            max-height: 85vh;
            border-radius: 20px; 
            box-shadow: 
                0 0 80px rgba(255, 68, 68, 0.3),
                0 20px 60px rgba(0, 0, 0, 0.8),
                inset 0 0 40px rgba(0, 0, 0, 0.5);
            border: 3px solid transparent;
            background: linear-gradient(45deg, rgba(255, 68, 68, 0.2), rgba(0, 204, 136, 0.2), rgba(255, 170, 0, 0.2)) border-box;
            transition: all 0.4s ease;
            position: relative;
        }

        .video-feed:hover {
            box-shadow: 
                0 0 120px rgba(255, 68, 68, 0.5),
                0 25px 80px rgba(0, 0, 0, 0.9);
            transform: scale(1.02);
        }

        .video-feed::before {
            content: '';
            position: absolute;
            top: -3px;
            left: -3px;
            right: -3px;
            bottom: -3px;
            background: linear-gradient(45deg, #ff4444, #00cc88, #ffaa00, #ff4444);
            border-radius: 23px;
            z-index: -1;
            animation: borderRotate 3s linear infinite;
        }

        @keyframes borderRotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .zone-item { 
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            margin-top: 12px; 
            padding: 16px; 
            border-radius: 10px; 
            font-size: 14px; 
            border-left: 4px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: hidden;
        }

        .zone-item:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            border-left-color: var(--accent-fire);
        }

        .zone-fire { 
            border-left-color: var(--accent-fire) !important;
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.15), rgba(183, 28, 28, 0.2)) !important;
            box-shadow: 0 0 20px rgba(255, 68, 68, 0.3) !important;
            animation: zonePulse 2s ease-in-out infinite;
        }

        @keyframes zonePulse {
            0%, 100% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.3); }
            50% { box-shadow: 0 0 30px rgba(255, 68, 68, 0.6); }
        }

        h3 {
            margin: 30px 0 20px 0;
            font-size: 16px; 
            color: var(--text-secondary);
            font-weight: 500;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .metric-icon {
            display: block;
            font-size: 28px;
            margin-bottom: 8px;
            opacity: 0.8;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .sidebar { width: 320px; }
            .stat-value { font-size: 20px; }
        }

        @media (max-width: 768px) {
            body { flex-direction: column; height: auto; }
            .sidebar { width: 100%; height: auto; border-right: none; border-bottom: 1px solid rgba(255,255,255,0.1); }
            .main-view { height: 70vh; }
        }

        /* Loading animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: var(--accent-fire);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
    <script>
        function updateData() {
            fetch('/api/stats').then(r => r.json()).then(data => {
                document.getElementById('status').innerText = data.status;
                document.getElementById('status').className = 'status-box ' + data.status;
                document.getElementById('temp').innerText = data.sensor.temp + '°C';
                document.getElementById('gas').innerText = data.sensor.gas;
                document.getElementById('ppl').innerText = data.people;
                document.getElementById('flame').innerText = data.sensor.fire == 1 ? "YES" : "NO";
                
                let zoneHtml = '';
                for (const [id, info] of Object.entries(data.zones)) {
                    let fireClass = info.fire ? 'zone-fire' : '';
                    zoneHtml += `<div class="zone-item ${fireClass}"><b>Zone ${id}:</b> ${info.people} People | ${info.trend}</div>`;
                }
                document.getElementById('zone-list').innerHTML = zoneHtml;
            }).catch(e => {
                console.log('Update failed:', e);
            });
        }
        setInterval(updateData, 1000);
        updateData();
    </script>
</head>
<body>
    <div class="sidebar">
        <div class="logo">
            <i class="fas fa-fire-flame-curved"></i>
            <h2>PYROGUARD V4</h2>
        </div>
        <div id="status" class="status-box SAFE">SYSTEM STARTING <span class="loading"></span></div>
        <div class="stat-grid">
            <div class="stat-card">
                <i class="fas fa-thermometer-half metric-icon"></i>
                <div class="stat-label">Temperature</div>
                <div class="stat-value" id="temp">--</div>
            </div>
            <div class="stat-card">
                <i class="fas fa-smog metric-icon"></i>
                <div class="stat-label">Gas Level</div>
                <div class="stat-value" id="gas">--</div>
            </div>
            <div class="stat-card">
                <i class="fas fa-users metric-icon"></i>
                <div class="stat-label">People Count</div>
                <div class="stat-value" id="ppl">--</div>
            </div>
            <div class="stat-card">
                <i class="fas fa-fire metric-icon"></i>
                <div class="stat-label">Flame Sensor</div>
                <div class="stat-value" id="flame">--</div>
            </div>
        </div>
        <h3><i class="fas fa-map-marked-alt"></i> ZONE MONITORING</h3>
        <div id="zone-list"></div>
    </div>
    <div class="main-view">
        <img src="/video_feed" class="video-feed" alt="Live Video Feed">
    </div>

</body>
</html>
"""

# =========================================================
# FLASK INFRASTRUCTURE
# =========================================================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/stats')
def get_stats():
    with state_lock:
        return jsonify({
            "status": shared_state["status"],
            "sensor": shared_state["sensor_data"],
            "people": shared_state["total_people"],
            "zones": shared_state["zone_info"]
        })

@app.route('/data', methods=['POST'])
def data_endpoint():
    d = request.get_json()
    with state_lock:
        if d: shared_state["sensor_data"].update(d)
        return jsonify({"status": shared_state["status"], "people": shared_state["total_people"]})

@app.route('/health')
def health(): return jsonify({"status": "ok"}), 200

def generate_mjpeg():
    global main_view
    while is_running:
        if main_view is not None:
            _, buffer = cv2.imencode('.jpg', main_view)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================================================
# WORKERS
# =========================================================
def stream_reader():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    while is_running:
        ret, frame = cap.read()
        if ret: latest_frame = frame
    cap.release()

def fire_ai_processor():
    global latest_predictions
    while is_running:
        if latest_frame is not None:
            results = fire_model.infer(image=latest_frame)
            latest_predictions = results[0].predictions
        time.sleep(0.1)

def mouse_handler(event, x, y, flags, params):
    global current_pts, aisle_configs
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append((x, y))
        if len(current_pts) == 4:
            poly = np.array(current_pts, np.int32)
            aisle_configs.append({"id": len(aisle_configs) + 1, "poly": poly})
            current_pts = []

# =========================================================
# MAIN
# =========================================================
Thread(target=stream_reader, daemon=True).start()
time.sleep(2)

print("\n1. SETUP MODE")
print("2. PYROGUARD V4 + DASHBOARD")
choice = input("Select Mode: ")

if choice == "1":
    cv2.namedWindow("Setup")
    cv2.setMouseCallback("Setup", mouse_handler)
    while True:
        if latest_frame is None: continue
        display = latest_frame.copy()
        for zone in aisle_configs:
            cv2.polylines(display, [zone["poly"]], True, (0, 255, 0), 2)
        for pt in current_pts:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        cv2.imshow("Setup", display)
        if cv2.waitKey(1) == ord('s'):
            np.save(CONFIG_FILE, aisle_configs)
            break

elif choice == "2":
    if not os.path.exists(CONFIG_FILE):
        print("❌ Setup required."); exit()
    
    aisle_configs = np.load(CONFIG_FILE, allow_pickle=True)
    Thread(target=fire_ai_processor, daemon=True).start()
    Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")

    while True:
        if latest_frame is None: continue
        
        main_view = latest_frame.copy()
        
        person_results = person_model(latest_frame, classes=[0], conf=0.5, verbose=False)[0]
        
        frame_people_count = 0
        current_active_fire_zones = {}
        zone_info_frame = {z["id"]: {"people": 0, "fire": False, "trend": "STABLE"} for z in aisle_configs}
        
        ai_fire_confirmed = False
        ai_escalating = False

        # --- People Logic ---
        for box in person_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for zone in aisle_configs:
                if cv2.pointPolygonTest(zone["poly"], (cx, cy), False) >= 0:
                    zone_info_frame[zone["id"]]["people"] += 1
                    frame_people_count += 1
                    cv2.rectangle(main_view, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # --- Fire Logic ---
        for pred in latest_predictions:
            if pred.class_name.lower() != "fire" or pred.confidence < CONF_THRESHOLD: continue
            center = (int(pred.x), int(pred.y))
            area = pred.width * pred.height
            
            for zone in aisle_configs:
                if cv2.pointPolygonTest(zone["poly"], center, False) >= 0:
                    zid = zone["id"]
                    zone_info_frame[zid]["fire"] = True
                    current_active_fire_zones[zid] = area
                    
                    if zid not in zone_timers: zone_timers[zid] = time.time()
                    confirmed = (time.time() - zone_timers[zid]) >= PERSISTENCE_THRESHOLD
                    if confirmed: ai_fire_confirmed = True

                    if zid not in zone_area_history: zone_area_history[zid] = []
                    zone_area_history[zid].append(area)
                    if len(zone_area_history[zid]) > TREND_WINDOW: zone_area_history[zid].pop(0)
                    
                    trend_text = "STABLE"
                    if confirmed and len(zone_area_history[zid]) > 5:
                        diff = zone_area_history[zid][-1] - zone_area_history[zid][0]
                        if diff > 500: trend_text = "ESCALATING 📈"; ai_escalating = True
                        elif diff < -500: trend_text = "REDUCING 📉"
                    
                    zone_info_frame[zid]["trend"] = trend_text
                    color = (0, 0, 255) if confirmed else (0, 165, 255)
                    cv2.rectangle(main_view, (int(pred.x - pred.width/2), int(pred.y - pred.height/2)), 
                                 (int(pred.x + pred.width/2), int(pred.y + pred.height/2)), color, 3)
                    

        # --- Sync with Shared State ---
        with state_lock:
            shared_state["total_people"] = frame_people_count
            shared_state["zone_info"] = zone_info_frame
            s = shared_state["sensor_data"]
            gas_warn = s['gas'] > GAS_THRESHOLD
            flame_sens = s['fire'] == 1 # Based on common active-low sensors
            temp_warn = s['temp'] > shared_state["temp_threshold"]

            if ai_fire_confirmed or ai_escalating or (gas_warn and flame_sens):
                shared_state["status"] = "FIRE"
            elif temp_warn or gas_warn or flame_sens:
                shared_state["status"] = "WARNING"
            else:
                shared_state["status"] = "SAFE"

        cv2.imshow("Intelligence Hub Running...", main_view)
        
        if cv2.waitKey(1) == ord('q'): break

is_running = False
cv2.destroyAllWindows()