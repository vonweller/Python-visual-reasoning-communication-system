import cv2
from PySide6.QtCore import QThread, Signal
from core.inference import YoloInference

class VideoThread(QThread):
    frame_processed = Signal(object, object) # annotated_frame, detections
    connection_status = Signal(bool, str) # success, message

    def __init__(self, camera_id=0, model_path="yolov8n.pt", conf_threshold=0.5, classes_dict=None, device="cpu"):
        super().__init__()
        self.camera_id = camera_id
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes_dict = classes_dict
        self.device = device
        self.running = False

    def run(self):
        self.running = True
        # Initialize YOLO in the thread
        yolo = YoloInference(self.model_path, self.conf_threshold, self.classes_dict, self.device)
        
        self.connection_status.emit(False, "正在连接...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            self.connection_status.emit(False, "无法连接到视频源")
            self.running = False
            return

        self.connection_status.emit(True, "已连接")
        
        # Optimize camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            ret, frame = cap.read()
            if ret:
                # Run inference
                detections, annotated, _ = yolo.predict(frame)
                self.frame_processed.emit(annotated, detections)
            else:
                self.msleep(100)
            
            # No sleep needed here, inference takes time, so it self-regulates. 
            # But to prevent 100% CPU on fast inference, maybe a tiny sleep if needed.
            # Actually, cap.read() blocks until frame is available usually.
            
        cap.release()

    def stop(self):
        self.running = False
        self.wait()
