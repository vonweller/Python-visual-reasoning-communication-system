import cv2
import os
from PySide6.QtCore import QThread, Signal
from core.inference import YoloInference

class BatchInferenceThread(QThread):
    progress_updated = Signal(int, int, str)
    result_ready = Signal(str, object, object, list)
    batch_finished = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, image_paths, model_path, conf_threshold, classes_dict, device="cpu"):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes_dict = classes_dict
        self.device = device
        self.running = True
        self.results = []

    def run(self):
        try:
            self.yolo = YoloInference(self.model_path, self.conf_threshold, self.classes_dict, self.device)
            total = len(self.image_paths)
            
            for idx, path in enumerate(self.image_paths):
                if not self.running:
                    break
                
                try:
                    filename = os.path.basename(path)
                    self.progress_updated.emit(idx + 1, total, f"正在处理: {filename}")
                    
                    img = cv2.imread(path)
                    if img is None:
                        self.error_occurred.emit(f"无法读取图片: {filename}")
                        continue
                    
                    detections, annotated, inference_time = self.yolo.predict(img)
                    
                    result = {
                        'path': path,
                        'filename': filename,
                        'original_image': img,
                        'annotated_image': annotated,
                        'detections': detections,
                        'inference_time': inference_time
                    }
                    self.results.append(result)
                    
                    self.result_ready.emit(filename, img, annotated, detections)
                    
                except Exception as e:
                    self.error_occurred.emit(f"处理图片 {os.path.basename(path)} 时出错: {str(e)}")
                    continue
            
            self.batch_finished.emit(len(self.results))
            
        except Exception as e:
            self.error_occurred.emit(f"批量推理初始化失败: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()

    def get_results(self):
        return self.results