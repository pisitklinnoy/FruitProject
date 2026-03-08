import cv2
import zmq
import json
import numpy as np
from ultralytics import YOLO

# ===================================================================
# ส่วนที่ 1: สมองกล AI (นี่คือส่วนที่ถูกย่อไว้ตรงคอมเมนต์ครับ)
# ===================================================================
class MangosteenAI:
    def __init__(self, detector_path, classifier_path):
        print("🚀 กำลังโหลดสมองกลทั้ง 2 ตัวเข้าสู่ระบบ...")
        self.detector = YOLO(detector_path)
        self.classifier = YOLO(classifier_path)
        self.grade_memory = {} # สมุดจดจำเกรดเพื่อลดภาระการ์ดจอ

    def process_frame(self, frame):
        """รับภาพ 1 เฟรมเข้ามา -> ส่งคืนพิกัดและเกรดออกไป"""
        results = self.detector.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, verbose=False)
        output_data = [] 

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # ถ้าเป็น ID ใหม่ ให้ส่งไปแยกเกรด
                if track_id not in self.grade_memory:
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size > 0:
                        cls_results = self.classifier(crop_img, verbose=False)
                        top_class_idx = cls_results[0].probs.top1
                        grade_name = cls_results[0].names[top_class_idx]
                        self.grade_memory[track_id] = grade_name
                
                # ดึงเกรดจากสมุดจด
                grade = self.grade_memory.get(track_id, "Unknown")

                # แพ็กข้อมูลใส่ตะกร้าเตรียมส่งกลับ
                output_data.append({
                    "id": int(track_id),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "grade": grade
                })

        return output_data

# ===================================================================
# ส่วนที่ 2: ระบบเซิร์ฟเวอร์เปิดท่อส่งข้อมูล (ZeroMQ)
# ===================================================================
def start_zmq_server():
    print("🧠 กำลังเตรียมเปิดท่อรับส่งข้อมูล...")
    
    # โหลดโมเดล (ตรวจสอบชื่อไฟล์ให้ตรงกับที่คุณเซฟไว้นะครับ)
    ai = MangosteenAI("best_mangosteen_yolo26s.pt", "grade_mangosteen_best.pt")

    # สร้างท่อ ZeroMQ แบบรอรับคำสั่ง (REP)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555") # เปิดพอร์ต 5555 รอกลางอากาศ

    print("🟢 Python AI Server เปิดทำงานแล้ว! รอรับรูปภาพจาก C++ ผ่านพอร์ต 5555...")

    while True:
        try:
            # 1. รอรับไฟล์ภาพที่ฝั่งหน้าบ้าน (C++) ส่งมา
            message = socket.recv()
            
            # 2. แปลงข้อมูลไบต์ให้กลับมาเป็นรูปภาพ
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 3. ส่งภาพเข้าสมองกล AI ด้านบน
            results = ai.process_frame(frame)

            # 4. แปลงผลลัพธ์เป็นข้อความ JSON แล้วส่งกลับลงท่อ
            json_data = json.dumps(results)
            socket.send_string(json_data)

        except Exception as e:
            print("เกิดข้อผิดพลาด:", e)
            # ถ้ามี Error ก็ส่ง Error กลับไป C++ จะได้ไม่ค้าง
            socket.send_string(json.dumps({"error": str(e)}))

# สั่งให้โปรแกรมเริ่มทำงาน
if __name__ == '__main__':
    start_zmq_server()