import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from PySide6.QtCore import QThread, Signal
import json
from core.inference import YoloInference

class MqttWorker(QThread):
    frame_processed = Signal(str, object, object) # topic, annotated_frame, detections
    connection_status = Signal(bool, str)

    def __init__(self, broker, port, topics, username=None, password=None, model_path="yolov8n.pt", conf_threshold=0.5, classes_dict=None, device="cpu"):
        super().__init__()
        self.broker = broker
        self.port = port
        self.topics = topics # list of dicts: [{"name": "...", "topic": "..."}]
        self.username = username
        self.password = password
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes_dict = classes_dict
        self.device = device
        self.client = mqtt.Client()
        self.running = False
        self.yolo = None

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def run(self):
        self.running = True
        # Initialize YOLO in the thread
        self.yolo = YoloInference(self.model_path, self.conf_threshold, self.classes_dict, self.device)
        
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            while self.running:
                self.msleep(100)
        except Exception as e:
            self.connection_status.emit(False, str(e))
        finally:
            self.client.loop_stop()

    def stop(self):
        self.running = False
        self.client.disconnect()
        self.wait()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connection_status.emit(True, "已连接")
            for t in self.topics:
                if isinstance(t, dict) and "topic" in t:
                    client.subscribe(t["topic"])
                elif isinstance(t, str):
                    client.subscribe(t)
        else:
            self.connection_status.emit(False, f"连接失败，代码 {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.connection_status.emit(False, "已断开")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            # Handle the specific format provided by user: "data:image/png;base64,..."
            if "base64," in payload:
                base64_data = payload.split("base64,")[1]
            else:
                base64_data = payload

            # Remove any trailing/leading whitespace or potential artifacts
            base64_data = base64_data.strip()
            
            # Fix padding if necessary (though usually not needed if string is correct)
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)

            img_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # Run inference
                if self.yolo:
                    detections, annotated, _ = self.yolo.predict(img)
                    self.frame_processed.emit(msg.topic, annotated, detections)
            else:
                print(f"Failed to decode image from topic {msg.topic}")

        except Exception as e:
            print(f"Error processing message on {msg.topic}: {e}")

    def publish_message(self, topic, payload):
        if self.client and self.client.is_connected():
            try:
                self.client.publish(topic, payload)
            except Exception as e:
                print(f"Publish error: {e}")
