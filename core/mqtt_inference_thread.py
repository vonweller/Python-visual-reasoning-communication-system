
import cv2
import numpy as np
import time
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from core.inference import YoloInference

class MqttInferenceThread(QThread):
    inference_finished = Signal(object, object)  # annotated_frame, detections
    error_occurred = Signal(str)

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, classes_dict=None, device="cpu"):
        super().__init__()
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes_dict = classes_dict
        self.device = device
        
        self.running = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.latest_frame_data = None
        self.frame_pending = False
        
        # Performance tuning
        self.last_inference_time = 0
        self.min_interval = 0.05  # Max 20 FPS to prevent CPU flooding, though thread will mostly overlap

    def set_config(self, conf_threshold, device):
        """Update runtime configuration"""
        self.conf_threshold = conf_threshold
        # Note: changing device requires re-initializing the model which is complex in running thread.
        # For now we only support updating confidence threshold easily. 
        # Device change usually requires restarting the app or the thread.

    def update_frame(self, image_bytes):
        """Thread-safe method to update the latest frame"""
        self.mutex.lock()
        self.latest_frame_data = image_bytes
        self.frame_pending = True
        # print(f"[MqttInferenceThread] Frame updated, len: {len(image_bytes)}") # Debug
        self.condition.wakeOne()  # Wake up the processing thread
        self.mutex.unlock()

    def run(self):
        self.running = True
        try:
            # Initialize YOLO instance in this thread
            yolo = YoloInference(self.model_path, self.conf_threshold, self.classes_dict, self.device)
            print(f"[MqttInferenceThread] Model initialized on {self.device}")
            
            while self.running:
                self.mutex.lock()
                # Wait for new frame
                while not self.frame_pending and self.running:
                    self.condition.wait(self.mutex)
                
                if not self.running:
                    self.mutex.unlock()
                    break
                
                # Get latest frame data
                frame_bytes = self.latest_frame_data
                self.frame_pending = False
                self.mutex.unlock()
                
                # Process frame
                if frame_bytes:
                    try:
                        t1 = time.time()
                        nparr = np.frombuffer(frame_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Verify confidence threshold
                            yolo.conf_threshold = self.conf_threshold
                            
                            # print(f"[MqttInferenceThread] Predicting... Shape: {frame.shape}")
                            detections, annotated, infer_time = yolo.predict(frame)
                            t2 = time.time()
                            print(f"[MqttInferenceThread] Inference done. Time: {infer_time:.1f}ms, Total: {(t2-t1)*1000:.1f}ms")
                            
                            self.inference_finished.emit(annotated, detections)
                        else:
                            print("[MqttInferenceThread] Frame decode failed (None)")
                            
                        # Small sleep to prevent absolutely zero idle time if flooding
                        # self.msleep(10) 
                        
                    except Exception as e:
                        self.error_occurred.emit(f"Inference error: {str(e)}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
             self.error_occurred.emit(f"Thread initialization error: {str(e)}")
        
        print("[MqttInferenceThread] Stopped")

    def stop(self):
        self.running = False
        self.mutex.lock()
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()
