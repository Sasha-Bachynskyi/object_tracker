import os
from datetime import datetime

from ultralytics import YOLO


def check_yolo_v8(file_path):
    start_time = datetime.now()
    model = YOLO(os.path.join(os.getcwd(), "yolov8s.pt"))
    model.predict(source=file_path, conf=0.25, show=True)
    print(f"Spent time: {datetime.now() - start_time}")


check_yolo_v8(os.path.join(os.getcwd(), "cut_traffic.mp4"))
