import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from PySide6.QtCore import QThread, Signal, QTimer
import json
from core.inference import YoloInference
import time

class MqttWorker(QThread):
    frame_processed = Signal(str, object, object)
    connection_status = Signal(bool, str)
    log_message = Signal(str)

    def __init__(self, broker, port, topics, username=None, password=None, model_path="yolov8n.pt", conf_threshold=0.5, classes_dict=None, device="cpu"):
        super().__init__()
        self.broker = broker
        self.port = port
        self.topics = topics
        self.username = username
        self.password = password
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes_dict = classes_dict
        self.device = device
        self.client = mqtt.Client()
        self.running = False
        self.yolo = None
        self.auto_reconnect = True
        self.reconnect_interval = 5
        self.connection_attempts = 0
        self.max_reconnect_attempts = 10

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def run(self):
        self.running = True
        self.yolo = YoloInference(self.model_path, self.conf_threshold, self.classes_dict, self.device)
        
        self.connection_attempts = 0
        while self.running:
            try:
                self.connection_attempts += 1
                self.log_message.emit(f"正在尝试连接 MQTT 服务器 {self.broker}:{self.port} (尝试 {self.connection_attempts})...")
                
                self.client.connect(self.broker, self.port, 60)
                self.client.loop_start()
                
                while self.running and self.client.is_connected():
                    self.msleep(100)
                
                if self.running and not self.client.is_connected():
                    self.client.loop_stop()
                    if self.auto_reconnect and self.connection_attempts < self.max_reconnect_attempts:
                        self.log_message.emit(f"连接断开，{self.reconnect_interval}秒后重连...")
                        self.msleep(self.reconnect_interval * 1000)
                        continue
                    elif self.connection_attempts >= self.max_reconnect_attempts:
                        self.log_message.emit(f"已达到最大重连次数 ({self.max_reconnect_attempts})，停止重连")
                        self.connection_status.emit(False, "连接失败")
                        break
                        
            except Exception as e:
                self.connection_status.emit(False, f"连接错误: {str(e)}")
                self.log_message.emit(f"连接异常: {str(e)}")
                
                if self.auto_reconnect and self.running and self.connection_attempts < self.max_reconnect_attempts:
                    self.log_message.emit(f"{self.reconnect_interval}秒后重连...")
                    self.msleep(self.reconnect_interval * 1000)
                elif self.connection_attempts >= self.max_reconnect_attempts:
                    self.log_message.emit(f"已达到最大重连次数 ({self.max_reconnect_attempts})，停止重连")
                    break
            finally:
                if not self.running:
                    self.client.loop_stop()

    def stop(self):
        self.running = False
        self.auto_reconnect = False
        if self.client.is_connected():
            self.client.disconnect()
        self.wait()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connection_attempts = 0
            self.connection_status.emit(True, "已连接")
            self.log_message.emit(f"MQTT 连接成功！订阅主题: {[t.get('topic', t) if isinstance(t, dict) else t for t in self.topics]}")
            
            for t in self.topics:
                if isinstance(t, dict) and "topic" in t:
                    client.subscribe(t["topic"])
                    self.log_message.emit(f"已订阅主题: {t['topic']}")
                elif isinstance(t, str):
                    client.subscribe(t)
                    self.log_message.emit(f"已订阅主题: {t}")
        else:
            self.connection_status.emit(False, f"连接失败，代码 {rc}")
            self.log_message.emit(f"MQTT 连接失败，错误代码: {rc}")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self.connection_status.emit(False, "已断开")
            self.log_message.emit(f"MQTT 连接断开，错误代码: {rc}")
        else:
            self.log_message.emit("MQTT 正常断开连接")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            if "base64," in payload:
                base64_data = payload.split("base64,")[1]
            else:
                base64_data = payload

            base64_data = base64_data.strip()
            
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)

            img_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                if self.yolo:
                    detections, annotated, _ = self.yolo.predict(img)
                    self.frame_processed.emit(msg.topic, annotated, detections)
            else:
                self.log_message.emit(f"无法解码主题 {msg.topic} 的图片数据")

        except Exception as e:
            self.log_message.emit(f"处理主题 {msg.topic} 的消息时出错: {str(e)}")

    def publish_message(self, topic, payload):
        if self.client and self.client.is_connected():
            try:
                result = self.client.publish(topic, payload)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self.log_message.emit(f"消息已发布到主题: {topic}")
                else:
                    self.log_message.emit(f"发布消息失败，错误代码: {result.rc}")
            except Exception as e:
                self.log_message.emit(f"发布消息异常: {str(e)}")
        else:
            self.log_message.emit("MQTT 未连接，无法发布消息")
