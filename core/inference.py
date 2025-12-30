from ultralytics import YOLO
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys

class YoloInference:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, classes_dict=None, device="cpu"):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.model_path = os.path.join(base_path, model_path)
        self.conf_threshold = conf_threshold
        self.classes_dict = classes_dict
        self.device = device
        self.model = None
        self.init_model()

    def init_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"[YoloInference] 模型已加载，使用设备: {self.device}")
        except Exception as e:
            if self.device != "cpu":
                raise Exception(f"GPU初始化失败: {str(e)}")
            else:
                raise e

    def set_device(self, device):
        self.device = device
        self.init_model()

    def predict(self, image):
        """
        Run inference on an image.
        Args:
            image: numpy array (cv2 image)
        Returns:
            results: list of detections
            annotated_frame: image with bounding boxes
            inference_time: time taken in ms
        """
        start_time = time.time()
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        # Convert to PIL for drawing Chinese
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Load Font (Try Microsoft YaHei)
        try:
            font = ImageFont.truetype("msyh.ttc", 20)
        except:
            try:
                font = ImageFont.truetype("simhei.ttf", 20)
            except:
                font = ImageFont.load_default()

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # Original English Name
                cls_name_en = self.model.names[cls_id]
                # Chinese Name Lookup
                cls_name_cn = "未知"
                if self.classes_dict and str(cls_id) in self.classes_dict:
                    cls_name_cn = self.classes_dict[str(cls_id)]
                
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                # Draw Box
                x1, y1, x2, y2 = map(int, xyxy)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                
                # Draw Label: English (Chinese)
                label = f"{cls_name_en} ({cls_name_cn}) {conf:.2f}"
                # Get text size
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw label background
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=(0, 255, 0))
                draw.text((x1 + 2, y1 - text_height - 4), label, fill=(0, 0, 0), font=font)

                detections.append({
                    "class_id": cls_id,
                    "class_name_en": cls_name_en,
                    "class_name_cn": cls_name_cn,
                    "confidence": conf,
                    "bbox": xyxy
                })

        # Convert back to cv2
        annotated_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return detections, annotated_frame, inference_time
