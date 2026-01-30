import cv2
import numpy as np
import os
import time
from threading import Thread, Lock
from flask import Flask, request, jsonify
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
# SHARED STATE (Thread-Safe Data Hub)
# =========================================================
state_lock = Lock()
shared_state = {
    "status": "SAFE",
    "sensor_data": {'temp': 0, 'hum': 0, 'gas': 0, 'fire': 1},
    "temp_threshold": 32,
    "total_people": 0,
    "zone_occupancy": {} # {zid: {"people": count, "fire": bool, "trend": str}}
}

aisle_configs = []
current_pts = []
latest_frame = None
latest_predictions = [] # Fire predictions from Roboflow
is_running = True
zone_timers = {}
zone_area_history = {}

# =========================================================
# AI MODELS
# =========================================================
os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY
fire_model = get_model(MODEL_ID)
person_model = YOLO("yolov8n.pt") # YOLOv8 for people

# =========================================================
# WORKER THREADS
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
            # Roboflow Fire Inference
            results = fire_model.infer(image=latest_frame)
            latest_predictions = results[0].predictions
        time.sleep(0.1)

# =========================================================
# FLASK SERVER (ESP32 Endpoint)
# =========================================================
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"}), 200

@app.route('/data', methods=['POST'])
def data():
    d = request.get_json()
    with state_lock:
        if d: shared_state["sensor_data"].update(d)
        # Optimized: Send status back in POST response
        return jsonify({
            "status": shared_state["status"], 
            "people": shared_state["total_people"]
        })

@app.route('/final_status', methods=['GET'])
def get_status(): return shared_state["status"]

# =========================================================
# MOUSE HANDLER
# =========================================================
def mouse_handler(event, x, y, flags, params):
    global current_pts, aisle_configs
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append((x, y))
        if len(current_pts) == 4:
            poly = np.array(current_pts, np.int32)
            aisle_configs.append({"id": len(aisle_configs) + 1, "poly": poly})
            current_pts = []

# =========================================================
# MAIN EXECUTION
# =========================================================
Thread(target=stream_reader, daemon=True).start()
time.sleep(2)

print("\n1. SETUP MODE (Draw Zones)")
print("2. PYROGUARD V4 (Intelligence Hub + People Detection)")
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
            print("💾 Zones Saved"); break

elif choice == "2":
    if not os.path.exists(CONFIG_FILE):
        print("❌ Run setup first."); exit()
    
    aisle_configs = np.load(CONFIG_FILE, allow_pickle=True)
    Thread(target=fire_ai_processor, daemon=True).start()
    Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()

    cv2.namedWindow("1. AI Intelligence Hub")
    cv2.namedWindow("2. Zone Reference Map")

    while True:
        if latest_frame is None: continue
        
        main_view = latest_frame.copy()
        zone_view = latest_frame.copy()
        
        # 1. RUN YOLO PERSON DETECTION (Class 0)
        person_results = person_model(latest_frame, classes=[0], conf=0.5, verbose=False)[0]
        
        # Reset current frame stats
        current_active_fire_zones = {}
        frame_people_count = 0
        zone_info = {z["id"]: {"people": 0, "fire": False, "trend": "STABLE"} for z in aisle_configs}
        
        ai_fire_confirmed = False
        ai_escalating = False

        # --- Process People Detections (FIXED TO COUNT ONLY IN ZONES) ---
        for box in person_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            person_in_zone = False
            # Identify which zone person is in
            for zone in aisle_configs:
                if cv2.pointPolygonTest(zone["poly"], (cx, cy), False) >= 0:
                    zone_info[zone["id"]]["people"] += 1
                    person_in_zone = True
            
            # ONLY increment global count and draw if inside a zone
            if person_in_zone:
                frame_people_count += 1
                cv2.rectangle(main_view, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan for People

        # --- Process Fire Detections (Roboflow) ---
        for pred in latest_predictions:
            if pred.class_name.lower() != "fire" or pred.confidence < CONF_THRESHOLD:
                continue

            center = (int(pred.x), int(pred.y))
            area = pred.width * pred.height
            
            for zone in aisle_configs:
                if cv2.pointPolygonTest(zone["poly"], center, False) >= 0:
                    zid = zone["id"]
                    zone_info[zid]["fire"] = True
                    current_active_fire_zones[zid] = area
                    
                    # Persistence Logic
                    if zid not in zone_timers: zone_timers[zid] = time.time()
                    elapsed = time.time() - zone_timers[zid]
                    confirmed = elapsed >= PERSISTENCE_THRESHOLD
                    if confirmed: ai_fire_confirmed = True

                    # Trend Logic
                    if zid not in zone_area_history: zone_area_history[zid] = []
                    zone_area_history[zid].append(area)
                    if len(zone_area_history[zid]) > TREND_WINDOW: zone_area_history[zid].pop(0)
                    
                    history = zone_area_history[zid]
                    trend_text = "STABLE"
                    t_color = (0, 165, 255) 
                    
                    if confirmed and len(history) > 5:
                        diff = history[-1] - history[0]
                        if diff > 500: 
                            trend_text = "ESCALATING 📈"; t_color = (0,0,255); ai_escalating = True
                        elif diff < -500: 
                            trend_text = "REDUCING 📉"; t_color = (0,255,0)
                    
                    zone_info[zid]["trend"] = trend_text

                    # Visuals
                    color = (0, 0, 255) if confirmed else (0, 165, 255)
                    x0, y0 = int(pred.x - pred.width/2), int(pred.y - pred.height/2)
                    x1, y1 = int(pred.x + pred.width/2), int(pred.y + pred.height/2)
                    cv2.rectangle(main_view, (x0, y0), (x1, y1), color, 3)
                    cv2.putText(main_view, f"FIRE | {trend_text}", (x0, y0 - 10), 1, 1, color, 2)
                    cv2.fillPoly(zone_view, [zone["poly"]], t_color)

        # Cleanup inactive fire zones
        for zid in list(zone_timers.keys()):
            if zid not in current_active_fire_zones:
                del zone_timers[zid]
                zone_area_history[zid] = []

        # --- FUSION LOGIC & SHARED STATE UPDATE ---
        with state_lock:
            s = shared_state["sensor_data"]
            shared_state["total_people"] = frame_people_count
            
            # Logic conditions
            gas_warn = s['gas'] > GAS_THRESHOLD
            flame_sens = s['fire'] == 0
            temp_warn = s['temp'] > shared_state["temp_threshold"]

            if ai_fire_confirmed or ai_escalating or (gas_warn and flame_sens):
                shared_state["status"] = "FIRE"
            elif temp_warn or gas_warn or flame_sens:
                shared_state["status"] = "WARNING"
            else:
                shared_state["status"] = "SAFE"

        # --- UI DASHBOARD ---
        cv2.rectangle(main_view, (10, 10), (600, 150), (0,0,0), -1)
        cv2.putText(main_view, f"SYSTEM: {shared_state['status']}", (25, 50), 2, 1.2, (255,255,255), 2)
        cv2.putText(main_view, f"PEOPLE: {frame_people_count} | T:{s['temp']}C | G:{s['gas']} | F:{s['fire']}", (25, 100), 2, 0.7, (0,255,0), 1)
        
        # Zone detail overlay
        y_off = 130
        for zid, info in zone_info.items():
            color = (0,0,255) if info["fire"] else (255,255,255)
            z_txt = f"Z{zid}: {info['people']} Pers | {'FIRE!' if info['fire'] else 'OK'}"
            cv2.putText(main_view, z_txt, (25, y_off), 2, 0.5, color, 1)
            y_off += 20
            cv2.polylines(zone_view, [aisle_configs[zid-1]["poly"]], True, (255,255,255), 2)

        cv2.imshow("1. AI Intelligence Hub", main_view)
        cv2.imshow("2. Zone Reference Map", zone_view)
        if cv2.waitKey(1) == ord('q'): break

is_running = False
cv2.destroyAllWindows()