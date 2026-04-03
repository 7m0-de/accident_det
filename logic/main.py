#Logic module
#الكود هذا يعطيك نتيجة اذا مقطع يحدد السياؤات الطبيعية والمصدمة ويعطي لها اللون وكذلك الامر مع الصور 
# يطبق على الكو لاب كما موضحة بخطوات
# -------------------------------------------------------------------خطوة الاولى للتثبيت
!pip install ultralytics

#خطوة الثانية لرفع مقطع او اي مدخل ----------------------------------------------------------
from google.colab import files
uploaded = files.upload()

#خطوة الثالثه الكود ودوالة الحسابية مع دالة اتخاذ القرار كاملة ------------------------------------------
from ultralytics import YOLO
import cv2
import time
from IPython.display import display, Image
import ipywidgets as widgets

# =====================ملحوظة قابلة للتطوير او تعديل=====================
# الإعدادات
# ==========================================
PROXIMITY_THRESHOLD = 80
STOP_THRESHOLD      = 0.5 # تم تعديل القيمة لتكون أكثر دقة في الكشف عن التوقف
LONG_STOP_SECONDS   = 5
CAR_CLASS_ID        = 2

previous_positions  = {}
stop_timers         = {}

# =====================قابلة للتعديل والاضافه=====================
# دوال الحساب
# ==========================================

def get_center(box):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return cx, cy

def get_distance(box1, box2):
    cx1, cy1 = get_center(box1)
    cx2, cy2 = get_center(box2)
    return ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5

def get_intersection(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    return max(0, ix2-ix1) * max(0, iy2-iy1)

def check_sudden_stop(obj_id, box):
    cx, cy = get_center(box)
    if obj_id in previous_positions:
        prev_cx, prev_cy = previous_positions[obj_id]
        dist = ((cx-prev_cx)**2 + (cy-prev_cy)**2) ** 0.5
        if dist < STOP_THRESHOLD:
            previous_positions[obj_id] = (cx, cy)
            return {
                "level"  : "MEDIUM",
                "type"   : "SUDDEN_STOP",
                "message": f"⚠ توقف مفاجئ → سيارة {obj_id}",
                "color"  : (0, 0, 255)
            }
    previous_positions[obj_id] = (cx, cy)
    return None

def check_long_stop(obj_id, box):
    cx, cy = get_center(box)
    if obj_id in previous_positions:
        prev_cx, prev_cy = previous_positions[obj_id]
        dist = ((cx-prev_cx)**2 + (cy-prev_cy)**2) ** 0.5
        if dist < STOP_THRESHOLD:
            if obj_id not in stop_timers:
                stop_timers[obj_id] = time.time()
            elapsed = time.time() - stop_timers[obj_id]
            if elapsed > LONG_STOP_SECONDS:
                return {
                    "level"  : "HIGH",
                    "type"   : "LONG_STOP",
                    "message": f"🔴 توقف طويل → سيارة {obj_id} واقفة {int(elapsed)} ثانية!",
                    "color"  : (0, 0, 200)
                }
        else:
            stop_timers.pop(obj_id, None)
    return None

def check_pair(id1, box1, id2, box2):
    intersection = get_intersection(box1, box2)
    distance     = get_distance(box1, box2)
    if intersection > 0:
        return {
            "level"  : "HIGH",
            "type"   : "COLLISION",
            "message": f"🔴 اصطدام → سيارة {id1} و سيارة {id2}",
            "color"  : (0, 0, 255)
        }
    elif distance < PROXIMITY_THRESHOLD:
        return {
            "level"  : "MEDIUM",
            "type"   : "DANGEROUS_PROXIMITY",
            "message": f"🟠 تقارب خطير → سيارة {id1} و سيارة {id2}",
            "color"  : (0, 165, 255)
        }
    else:
        return {
            "level"  : "NORMAL",
            "type"   : "NORMAL",
            "message": None,
            "color"  : (0, 255, 0)
        }

def analyze_frame(ids, coords, classes):
    decisions = []
    cars = []
    for i in range(len(ids)):
        if int(classes[i]) == CAR_CLASS_ID:
            cars.append({"id": int(ids[i]), "box": coords[i]})

    for car in cars:
        r1 = check_sudden_stop(car["id"], car["box"])
        if r1: decisions.append(r1)
        r2 = check_long_stop(car["id"], car["box"])
        if r2: decisions.append(r2)

    for i in range(len(cars)):
        for j in range(i+1, len(cars)):
            result = check_pair(
                cars[i]["id"], cars[i]["box"],
                cars[j]["id"], cars[j]["box"]
            )
            if result["level"] != "NORMAL":
                decisions.append(result)
    return decisions

def print_decision(decisions, frame_number):
    print(f"\n{'='*40}")
    print(f"📍 الفريم {frame_number}")
    if not decisions:
        print("✅ NORMAL — لا يوجد حادث")
        return
    priority = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    decisions.sort(key=lambda x: priority.get(x["level"], 3))
    for d in decisions:
        print(f"   [{d['level']}] {d['message']}")

def draw_results(frame, ids, coords, classes, decisions):
    for i in range(len(ids)):
        if int(classes[i]) != CAR_CLASS_ID:
            continue
        obj_id       = int(ids[i])
        x1,y1,x2,y2 = map(int, coords[i])
        color        = (0, 255, 0)
        for d in decisions:
            if str(obj_id) in d["message"]:
                color = d["color"]
                cv2.putText(frame, d["type"],
                            (x1, y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"ID {obj_id}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    return frame

# ===================يمكن ان اطورها او ايمن يساعدني بخبرته =======================
# الحلقة الرئيسية — مخصصة لـ 
# ==========================================

model        = YOLO("yolov8n.pt")
cap          = cv2.VideoCapture("video.mp4")
frame_number = 0

# إعداد حفظ الفيديو الناتج
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter('output.mp4', fourcc, 20.0,
         (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    results = model.track(frame, persist=True)
    boxes   = results[0].boxes

    if boxes.id is not None:
        ids     = boxes.id.cpu().numpy()
        coords  = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        decisions = analyze_frame(ids, coords, classes)
        print_decision(decisions, frame_number)
        frame = draw_results(frame, ids, coords, classes, decisions)

    # حفظ الفريم في الفيديو الناتج
    out.write(frame)

cap.release()
out.release()
print("\n✅ خلص! الفيديو محفوظ بـ output.mp4")

#خطوة الرابعة ------------------------------------------------------------
files.download('output.mp4')
