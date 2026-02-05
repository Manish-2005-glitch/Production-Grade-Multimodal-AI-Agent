from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect(frame):
    results = model(frame)
    
    out =[]
    
    for r in results:
        for b in r.boxes:
            out.append({
                "bbox": list(map(int, b.xyxy[0])),
                "conf": float(b.conf),
                "cls": int(b.cls)
            })
            
    return out