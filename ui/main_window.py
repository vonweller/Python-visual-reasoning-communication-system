import sys
import os
import cv2
import datetime
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QTabWidget, QFileDialog, QGroupBox, 
                               QFormLayout, QLineEdit, QSpinBox, QMessageBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QDoubleSpinBox,
                               QComboBox, QRadioButton, QButtonGroup)
from PySide6.QtCore import Slot, Qt
import json

from core.config_manager import ConfigManager
from core.inference import YoloInference
from core.mqtt_worker import MqttWorker
from core.mqtt_server import MqttServer
from core.video_thread import VideoThread
from core.batch_inference_thread import BatchInferenceThread
from core.mqtt_inference_thread import MqttInferenceThread
from ui.widgets import ImageDisplayWidget, LogTableWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load Config
        self.config_manager = ConfigManager()
        
        window_title = self.config_manager.get("ui.window_title", "Vonwell的MQTT科创本地云端摄像头推理面板BYQQ529538187")
        self.setWindowTitle(window_title)
        
        self.resize(1200, 800)

        # Initialize Yolo
        self.yolo = YoloInference(
            model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
            conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
            classes_dict=self.config_manager.classes,
            device=self.config_manager.get("yolo.device", "cpu")
        )

        # Workers
        self.mqtt_worker = None
        self.video_thread = None
        self.http_thread = None
        self.batch_inference_thread = None

        # Batch inference data
        self.batch_results = []
        self.current_batch_index = 0

        # MQTT Workers
        self.mqtt_worker = None
        self.mqtt_inference_thread = None  # Server mode inference thread

        # Predefined Themes
        self.color_themes = [
            ("科技蓝", "#007acc"),
            ("翡翠绿", "#28a745"),
            ("活力橙", "#fd7e14"),
            ("玫瑰红", "#dc3545"),
            ("深邃紫", "#6f42c1"),
            ("天空蓝", "#17a2b8"),
            ("柠檬黄", "#ffc107"),
            ("极简灰", "#6c757d"),
            ("暗夜黑", "#000000"),
            ("樱花粉", "#e83e8c")
        ]

        # UI Setup
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Local Image
        self.local_tab = QWidget()
        self.setup_local_tab()
        self.tabs.addTab(self.local_tab, "本地图片")

        # Tab 2: Camera
        self.camera_tab = QWidget()
        self.setup_camera_tab()
        self.tabs.addTab(self.camera_tab, "摄像头")

        # Tab 3: HTTP Camera
        self.http_tab = QWidget()
        self.setup_http_tab()
        self.tabs.addTab(self.http_tab, "HTTP 监控")

        # Tab 4: MQTT
        self.mqtt_tab = QWidget()
        self.setup_mqtt_tab()
        self.tabs.addTab(self.mqtt_tab, "MQTT 远程")

        # Tab 5: Settings
        self.settings_tab = QWidget()
        self.setup_settings_tab()
        self.tabs.addTab(self.settings_tab, "设置")

        # Data Log Area (Bottom)
        log_group = QGroupBox("推理日志")
        log_layout = QVBoxLayout(log_group)
        self.log_table = LogTableWidget()
        log_layout.addWidget(self.log_table)
        
        # Use Splitter to resize between tabs and log
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(log_group)
        
        # Set explicitly logical sizes to ensure log area is visible enough
        # assuming total height ~800. 500 for tabs, 300 for logs.
        splitter.setSizes([500, 300])
        
        main_layout.addWidget(splitter)

    def setup_local_tab(self):
        layout = QHBoxLayout(self.local_tab)
        
        controls_layout = QVBoxLayout()
        self.btn_load_image = QPushButton("加载图片")
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_folder = QPushButton("批量导入") 
        self.btn_load_folder.clicked.connect(self.load_folder)
        
        controls_layout.addWidget(self.btn_load_image)
        controls_layout.addWidget(self.btn_load_folder)
        
        # Batch inference controls
        batch_group = QGroupBox("批量推理")
        batch_layout = QVBoxLayout(batch_group)
        
        # Progress bar
        from PySide6.QtWidgets import QProgressBar
        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        self.batch_progress.setVisible(False)
        batch_layout.addWidget(self.batch_progress)
        
        # Progress label
        self.batch_progress_label = QLabel("准备就绪")
        self.batch_progress_label.setVisible(False)
        batch_layout.addWidget(self.batch_progress_label)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.btn_prev_result = QPushButton("上一张")
        self.btn_prev_result.clicked.connect(self.show_prev_batch_result)
        self.btn_prev_result.setEnabled(False)
        self.btn_next_result = QPushButton("下一张")
        self.btn_next_result.clicked.connect(self.show_next_batch_result)
        self.btn_next_result.setEnabled(False)
        
        nav_layout.addWidget(self.btn_prev_result)
        nav_layout.addWidget(self.btn_next_result)
        batch_layout.addLayout(nav_layout)
        
        # Result index label
        self.batch_index_label = QLabel("0 / 0")
        self.batch_index_label.setAlignment(Qt.AlignCenter)
        self.batch_index_label.setVisible(False)
        batch_layout.addWidget(self.batch_index_label)
        
        # Stop button
        self.btn_stop_batch = QPushButton("停止处理")
        self.btn_stop_batch.clicked.connect(self.stop_batch_inference)
        self.btn_stop_batch.setEnabled(False)
        batch_layout.addWidget(self.btn_stop_batch)
        
        controls_layout.addWidget(batch_group)
        controls_layout.addStretch()
        
        self.local_display_orig = ImageDisplayWidget("原始图像")
        self.local_display_res = ImageDisplayWidget("推理结果")
        
        layout.addLayout(controls_layout, 1)
        layout.addWidget(self.local_display_orig, 4)
        layout.addWidget(self.local_display_res, 4)

    def setup_camera_tab(self):
        layout = QVBoxLayout(self.camera_tab)
        
        controls_layout = QHBoxLayout()
        self.btn_start_cam = QPushButton("开启摄像头")
        self.btn_start_cam.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.btn_start_cam)
        controls_layout.addStretch()
        
        self.cam_display = ImageDisplayWidget("摄像头画面")
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.cam_display)

    def setup_http_tab(self):
        layout = QVBoxLayout(self.http_tab)
        
        controls_layout = QHBoxLayout()
        self.edit_http_url = QLineEdit(self.config_manager.get("yolo.http_stream_url", "http://192.168.1.32:81/stream"))
        self.edit_http_url.setPlaceholderText("输入 HTTP 视频流地址")
        
        self.btn_start_http = QPushButton("开启 HTTP 监控")
        self.btn_start_http.clicked.connect(self.toggle_http_camera)
        
        controls_layout.addWidget(QLabel("流地址:"))
        controls_layout.addWidget(self.edit_http_url, 1)
        controls_layout.addWidget(self.btn_start_http)
        
        self.http_display = ImageDisplayWidget("HTTP 监控画面")
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.http_display)

    def setup_mqtt_tab(self):
        layout = QVBoxLayout(self.mqtt_tab)
        
        controls_layout = QHBoxLayout()
        mqtt_mode = self.config_manager.get("mqtt.mode", "client")
        if mqtt_mode == "server":
            self.btn_connect_mqtt = QPushButton("启动 MQTT 服务端")
        else:
            self.btn_connect_mqtt = QPushButton("连接 MQTT")
        self.btn_connect_mqtt.clicked.connect(self.toggle_mqtt)
        self.lbl_mqtt_status = QPushButton("状态: 未连接")
        self.lbl_mqtt_status.setEnabled(False)
        self.lbl_mqtt_status.setStyleSheet("background-color: #555; color: white;")
        
        controls_layout.addWidget(self.btn_connect_mqtt)
        controls_layout.addWidget(self.lbl_mqtt_status)
        controls_layout.addStretch()
        
        # New Message Sending Area
        self.send_msg_widget = QWidget()
        send_layout = QHBoxLayout(self.send_msg_widget)
        send_layout.setContentsMargins(0, 5, 0, 5)
        
        self.edit_pub_topic = QLineEdit()
        self.edit_pub_topic.setPlaceholderText("Topic (例如: siot/舵机)")
        self.edit_pub_topic.setText("siot/舵机") # Default
        
        self.edit_pub_message = QLineEdit()
        self.edit_pub_message.setPlaceholderText("Message (例如: hello)")
        
        self.btn_pub_send = QPushButton("发送")
        self.btn_pub_send.clicked.connect(self.send_manual_mqtt_message)
        
        send_layout.addWidget(QLabel("发布消息:"))
        send_layout.addWidget(self.edit_pub_topic, 1) # Ratio 1
        send_layout.addWidget(self.edit_pub_message, 2) # Ratio 2
        send_layout.addWidget(self.btn_pub_send)
        
        # Initially only visible if needed, or always visible? 
        # I'll make it visible but disabled if not connected, or just let it exist.
        # Let's simple add it to layout.
        
        self.mqtt_display = ImageDisplayWidget("MQTT 画面")
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.send_msg_widget)
        layout.addWidget(self.mqtt_display)
        
        self.mqtt_log_text = QTableWidget()
        self.mqtt_log_text.setColumnCount(2)
        self.mqtt_log_text.setHorizontalHeaderLabels(["时间", "消息"])
        self.mqtt_log_text.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mqtt_log_text.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.mqtt_log_text.verticalHeader().setVisible(False)
        self.mqtt_log_text.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mqtt_log_text.setMaximumHeight(150)
        
        layout.addWidget(self.mqtt_log_text)
        
        self.mqtt_log_counter = 0
        
        # Initialize MQTT server
        self.mqtt_server = None

    def setup_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        
        # MQTT Settings Group
        mqtt_group = QGroupBox("MQTT 设置")
        mqtt_layout = QVBoxLayout(mqtt_group)
        
        # MQTT Mode Selection (Server/Client)
        mode_layout = QHBoxLayout()
        self.radio_mqtt_client = QRadioButton("客户端模式 (连接到服务器)")
        self.radio_mqtt_server = QRadioButton("服务端模式 (作为MQTT Broker)")
        
        self.mqtt_mode_button_group = QButtonGroup()
        self.mqtt_mode_button_group.addButton(self.radio_mqtt_client, 0)
        self.mqtt_mode_button_group.addButton(self.radio_mqtt_server, 1)
        
        current_mqtt_mode = self.config_manager.get("mqtt.mode", "client")
        if current_mqtt_mode == "server":
            self.radio_mqtt_server.setChecked(True)
        else:
            self.radio_mqtt_client.setChecked(True)
        
        self.radio_mqtt_client.toggled.connect(self.on_mqtt_mode_changed)
        self.radio_mqtt_server.toggled.connect(self.on_mqtt_mode_changed)
        
        mode_layout.addWidget(self.radio_mqtt_client)
        mode_layout.addWidget(self.radio_mqtt_server)
        mode_layout.addStretch()
        mqtt_layout.addLayout(mode_layout)
        
        # Client Mode Settings
        self.client_settings_widget = QWidget()
        client_form_layout = QFormLayout(self.client_settings_widget)
        
        # MQTT Server Presets
        mqtt_server_layout = QHBoxLayout()
        self.combo_mqtt_server = QComboBox()
        self.combo_mqtt_server.addItem("自定义", "custom")
        self.combo_mqtt_server.addItem("本地服务器 (127.0.0.1)", "local")
        self.combo_mqtt_server.addItem("远程服务器 (10.1.2.3)", "remote")
        self.combo_mqtt_server.currentIndexChanged.connect(self.on_mqtt_server_changed)
        
        self.btn_test_mqtt = QPushButton("测试连接")
        self.btn_test_mqtt.clicked.connect(self.test_mqtt_connection)
        
        mqtt_server_layout.addWidget(QLabel("服务器预设:"))
        mqtt_server_layout.addWidget(self.combo_mqtt_server, 1)
        mqtt_server_layout.addWidget(self.btn_test_mqtt)
        
        client_form_layout.addRow(mqtt_server_layout)
        
        self.edit_broker = QLineEdit(self.config_manager.get("mqtt.broker"))
        self.edit_port = QSpinBox()
        self.edit_port.setRange(1, 65535)
        self.edit_port.setValue(self.config_manager.get("mqtt.port"))
        
        self.edit_user = QLineEdit(self.config_manager.get("mqtt.username", ""))
        self.edit_pass = QLineEdit(self.config_manager.get("mqtt.password", ""))
        self.edit_pass.setEchoMode(QLineEdit.Password)
        
        client_form_layout.addRow("MQTT 服务器:", self.edit_broker)
        client_form_layout.addRow("MQTT 端口:", self.edit_port)
        client_form_layout.addRow("MQTT 用户名:", self.edit_user)
        client_form_layout.addRow("MQTT 密码:", self.edit_pass)
        
        mqtt_layout.addWidget(self.client_settings_widget)
        
        # Server Mode Settings
        self.server_settings_widget = QWidget()
        server_form_layout = QFormLayout(self.server_settings_widget)
        
        self.edit_server_port = QSpinBox()
        self.edit_server_port.setRange(1, 65535)
        self.edit_server_port.setValue(self.config_manager.get("mqtt.server_port", 1883))
        
        self.edit_server_host = QLineEdit(self.config_manager.get("mqtt.server_host", "0.0.0.0"))
        
        server_form_layout.addRow("监听地址:", self.edit_server_host)
        server_form_layout.addRow("监听端口:", self.edit_server_port)
        
        mqtt_layout.addWidget(self.server_settings_widget)
        
        # Show/hide settings based on mode
        self.on_mqtt_mode_changed()
        
        layout.addWidget(mqtt_group)
        
        # Inference Settings
        inf_group = QGroupBox("推理设置")
        inf_layout = QFormLayout(inf_group)
        
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(self.config_manager.get("yolo.conf_threshold", 0.5))
        
        inf_layout.addRow("置信度阈值:", self.spin_conf)
        
        self.edit_model_name = QLineEdit()
        self.edit_model_name.setPlaceholderText("例如: best.pt")
        self.edit_model_name.setText(self.config_manager.get("yolo.model_path", "best.pt"))
        inf_layout.addRow("模型名称:", self.edit_model_name)
        
        # Device Selection (GPU/CPU)
        device_group = QGroupBox("硬件加速")
        device_layout = QVBoxLayout(device_group)
        
        # CPU option
        cpu_layout = QHBoxLayout()
        self.radio_cpu = QRadioButton("CPU运行")
        self.lbl_cpu_check = QLabel("✓")
        self.lbl_cpu_check.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        self.lbl_cpu_check.setVisible(False)
        cpu_layout.addWidget(self.radio_cpu)
        cpu_layout.addWidget(self.lbl_cpu_check)
        cpu_layout.addStretch()
        
        # GPU option
        gpu_layout = QHBoxLayout()
        self.radio_gpu = QRadioButton("GPU加速")
        self.lbl_gpu_check = QLabel("✓")
        self.lbl_gpu_check.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        self.lbl_gpu_check.setVisible(False)
        gpu_layout.addWidget(self.radio_gpu)
        gpu_layout.addWidget(self.lbl_gpu_check)
        gpu_layout.addStretch()
        
        self.device_button_group = QButtonGroup()
        self.device_button_group.addButton(self.radio_cpu, 0)
        self.device_button_group.addButton(self.radio_gpu, 1)
        
        current_device = self.config_manager.get("yolo.device", "cpu")
        if current_device == "cuda" or current_device == "gpu":
            self.radio_gpu.setChecked(True)
            self.lbl_gpu_check.setVisible(True)
        else:
            self.radio_cpu.setChecked(True)
            self.lbl_cpu_check.setVisible(True)
        
        # Connect radio button signals for real-time check mark update
        self.radio_cpu.toggled.connect(self.update_device_check_mark)
        self.radio_gpu.toggled.connect(self.update_device_check_mark)
        
        device_layout.addLayout(cpu_layout)
        device_layout.addLayout(gpu_layout)
        
        inf_layout.addRow("设备选择:", device_group)
        layout.addWidget(inf_group)

        # UI Settings
        ui_group = QGroupBox("界面设置")
        ui_layout = QFormLayout(ui_group)
        
        # Theme Mode
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Dark", "Light"])
        self.combo_mode.setItemText(0, "深色模式")
        self.combo_mode.setItemText(1, "浅色模式")
        current_mode = self.config_manager.get("ui.theme", "dark")
        self.combo_mode.setCurrentIndex(0 if current_mode == "dark" else 1)
        
        # Window Title
        self.edit_window_title = QLineEdit()
        self.edit_window_title.setText(self.config_manager.get("ui.window_title", "Vonwell的MQTT科创本地云端摄像头推理面板BYQQ529538187"))
        self.edit_window_title.setPlaceholderText("请输入窗口标题")
        
        # Theme Color
        self.combo_color = QComboBox()
        current_color = self.config_manager.get("ui.theme_color", "#007acc")
        
        for name, color in self.color_themes:
            self.combo_color.addItem(name, color)
            
        # Set current index
        index = self.combo_color.findData(current_color)
        if index >= 0:
            self.combo_color.setCurrentIndex(index)
        else:
            self.combo_color.setCurrentIndex(0)
        
        ui_layout.addRow("界面模式:", self.combo_mode)
        ui_layout.addRow("主题颜色:", self.combo_color)
        ui_layout.addRow("窗口标题:", self.edit_window_title)
        ui_layout.addRow("主题颜色:", self.combo_color)
        layout.addWidget(ui_group)
        
        # Topic Management
        topic_group = QGroupBox("主题管理")
        topic_layout = QVBoxLayout(topic_group)
        
        self.topic_table = QTableWidget()
        self.topic_table.setColumnCount(2)
        self.topic_table.setHorizontalHeaderLabels(["名称", "主题"])
        self.topic_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.topic_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Load topics
        topics = self.config_manager.get("mqtt.topics", [])
        self.topic_table.setRowCount(len(topics))
        for i, t in enumerate(topics):
            # Handle both old dict format (if any) and new list format
            if isinstance(t, dict):
                self.topic_table.setItem(i, 0, QTableWidgetItem(t.get("name", "")))
                self.topic_table.setItem(i, 1, QTableWidgetItem(t.get("topic", "")))
            
        topic_layout.addWidget(self.topic_table)
        
        btn_layout = QHBoxLayout()
        self.btn_add_topic = QPushButton("添加主题")
        self.btn_add_topic.clicked.connect(self.add_topic)
        self.btn_del_topic = QPushButton("删除主题")
        self.btn_del_topic.clicked.connect(self.del_topic)
        
        btn_layout.addWidget(self.btn_add_topic)
        btn_layout.addWidget(self.btn_del_topic)
        topic_layout.addLayout(btn_layout)
        
        layout.addWidget(topic_group)

        self.btn_save_config = QPushButton("保存配置")
        self.btn_save_config.clicked.connect(self.save_settings)
        layout.addWidget(self.btn_save_config)

    def add_topic(self):
        row = self.topic_table.rowCount()
        self.topic_table.insertRow(row)
        self.topic_table.setItem(row, 0, QTableWidgetItem("新主题"))
        self.topic_table.setItem(row, 1, QTableWidgetItem("siot/new"))

    def del_topic(self):
        row = self.topic_table.currentRow()
        if row >= 0:
            self.topic_table.removeRow(row)

    def apply_styles(self):
        theme_color = self.config_manager.get("ui.theme_color", "#007acc")
        theme_mode = self.config_manager.get("ui.theme", "dark")
        
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        qss_file = os.path.join(base_path, "ui/styles.qss")
        if theme_mode == "light":
            qss_file = os.path.join(base_path, "ui/styles_light.qss")
            
        try:
            if os.path.exists(qss_file):
                with open(qss_file, "r") as f:
                    style = f.read()
                    # Replace default accent color with configured color
                    style = style.replace("#007acc", theme_color)
                    self.setStyleSheet(style)
            else:
                print(f"Style file not found: {qss_file}")
        except Exception as e:
            print(f"Error loading styles: {e}")

    # --- Logic ---

    def log_result(self, source, detections):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for d in detections:
            self.log_table.add_record(timestamp, source, d['class_name_en'], d['class_name_cn'], d['confidence'])
        
        # Publish to MQTT if connected
        if self.mqtt_worker and self.mqtt_worker.isRunning():
            publish_topic = self.config_manager.get("mqtt.publish_topic", "siot/推理结果")
            
            # Simplified payload: just the Chinese class names
            # If multiple detections, join them with comma
            class_names = [d['class_name_cn'] for d in detections]
            payload_str = ",".join(class_names)
            
            self.mqtt_worker.publish_message(publish_topic, payload_str)
        
        # Publish to MQTT Server if running (Server Mode)
        if self.mqtt_server and self.mqtt_server.is_running():
            publish_topic = self.config_manager.get("mqtt.publish_topic", "siot/推理结果")
            
            # Simplified payload: just the Chinese class names
            class_names = [d['class_name_cn'] for d in detections]
            payload_str = ",".join(class_names)
            
            # The server will broadcast this to all subscribers of the topic
            self.mqtt_server.publish_message(publish_topic, payload_str)
            print(f"[MainWindow] Published to MQTT Server topic '{publish_topic}': {payload_str}")

    # Local Image
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.process_local_image(path)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片文件")
                return
            
            self.start_batch_inference(files)

    def process_local_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return
        
        self.local_display_orig.update_image(img)
        detections, annotated, _ = self.yolo.predict(img)
        self.local_display_res.update_image(annotated)
        self.log_result("本地图片", detections)

    # Camera
    def toggle_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.btn_start_cam.setText("开启摄像头")
        else:
            self.video_thread = VideoThread(
                model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
                conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
                classes_dict=self.config_manager.classes,
                device=self.config_manager.get("yolo.device", "cpu")
            )
            self.video_thread.frame_processed.connect(self.process_camera_result)
            self.video_thread.connection_status.connect(self.on_camera_status)
            self.video_thread.start()
            self.btn_start_cam.setText("正在连接...")
            self.btn_start_cam.setEnabled(False)

    def on_camera_status(self, success, message):
        if success:
            self.btn_start_cam.setText("关闭摄像头")
            self.btn_start_cam.setEnabled(True)
        else:
            if message == "正在连接...":
                return
            QMessageBox.warning(self, "错误", f"摄像头错误: {message}")
            self.btn_start_cam.setText("开启摄像头")
            self.btn_start_cam.setEnabled(True)
            self.video_thread = None

    def process_camera_result(self, annotated_frame, detections):
        self.cam_display.update_image(annotated_frame)
        if detections:
            self.log_result("摄像头", detections)
        
        if self.mqtt_worker and self.mqtt_worker.isRunning():
            try:
                import base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                self.mqtt_worker.publish_message("siot/摄像头", img_base64)
            except Exception as e:
                pass

    # HTTP Camera
    def toggle_http_camera(self):
        if self.http_thread and self.http_thread.isRunning():
            self.http_thread.stop()
            self.btn_start_http.setText("开启 HTTP 监控")
            self.edit_http_url.setEnabled(True)
        else:
            url = self.edit_http_url.text().strip()
            if not url:
                QMessageBox.warning(self, "错误", "请输入有效的 HTTP 流地址")
                return
                
            # Save config
            self.config_manager.set("yolo.http_stream_url", url)
            
            self.http_thread = VideoThread(
                camera_id=url,
                model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
                conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
                classes_dict=self.config_manager.classes,
                device=self.config_manager.get("yolo.device", "cpu")
            )
            self.http_thread.frame_processed.connect(self.process_http_result)
            self.http_thread.connection_status.connect(self.on_http_status)
            self.http_thread.start()
            self.btn_start_http.setText("正在连接...")
            self.btn_start_http.setEnabled(False)
            self.edit_http_url.setEnabled(False)

    def on_http_status(self, success, message):
        if success:
            self.btn_start_http.setText("关闭 HTTP 监控")
            self.btn_start_http.setEnabled(True)
        else:
            if message == "正在连接...":
                return
            QMessageBox.warning(self, "错误", f"HTTP 连接错误: {message}")
            self.btn_start_http.setText("开启 HTTP 监控")
            self.btn_start_http.setEnabled(True)
            self.edit_http_url.setEnabled(True)
            self.http_thread = None

    def process_http_result(self, annotated_frame, detections):
        self.http_display.update_image(annotated_frame)
        if detections:
            self.log_result("HTTP 监控", detections)

    # MQTT
    def toggle_mqtt(self):
        mqtt_mode = self.config_manager.get("mqtt.mode", "client")
        
        if mqtt_mode == "server":
            if self.mqtt_server and self.mqtt_server.is_running():
                self.mqtt_server.stop()
                if self.mqtt_inference_thread:
                    self.mqtt_inference_thread.stop()
                self.btn_connect_mqtt.setText("启动 MQTT 服务端")
                self.lbl_mqtt_status.setText("状态: 已停止")
                self.lbl_mqtt_status.setStyleSheet("background-color: #555; color: white;")
            else:
                host = self.config_manager.get("mqtt.server_host", "0.0.0.0")
                port = self.config_manager.get("mqtt.server_port", 1883)
                
                # Start Inference Thread
                self.mqtt_inference_thread = MqttInferenceThread(
                    model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
                    conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
                    classes_dict=self.config_manager.classes,
                    device=self.config_manager.get("yolo.device", "cpu")
                )
                self.mqtt_inference_thread.inference_finished.connect(self.on_mqtt_inference_finished)
                self.mqtt_inference_thread.error_occurred.connect(lambda err: self.log_mqtt_message(f"推理错误: {err}"))
                self.mqtt_inference_thread.start()

                self.mqtt_server = MqttServer(host=host, port=port)
                self.mqtt_server.server_started.connect(self.on_mqtt_server_started)
                self.mqtt_server.server_stopped.connect(self.on_mqtt_server_stopped)
                self.mqtt_server.client_connected.connect(self.on_mqtt_client_connected)
                self.mqtt_server.client_disconnected.connect(self.on_mqtt_client_disconnected)
                self.mqtt_server.message_received.connect(self.on_mqtt_server_message)
                self.mqtt_server.image_data_received.connect(self.on_mqtt_server_image_data)
                self.mqtt_server.log_message.connect(self.log_mqtt_message)
                self.mqtt_server.start()
                self.btn_connect_mqtt.setText("正在启动...")
        else:
            if self.mqtt_worker and self.mqtt_worker.isRunning():
                self.mqtt_worker.stop()
                self.btn_connect_mqtt.setText("连接 MQTT")
                self.lbl_mqtt_status.setText("状态: 未连接")
                self.lbl_mqtt_status.setStyleSheet("background-color: #555; color: white;")
            else:
                broker = self.config_manager.get("mqtt.broker")
                port = self.config_manager.get("mqtt.port")
                topics = self.config_manager.get("mqtt.topics")
                username = self.config_manager.get("mqtt.username")
                password = self.config_manager.get("mqtt.password")
                
                self.mqtt_worker = MqttWorker(
                    broker, port, topics, username, password,
                    model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
                    conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
                    classes_dict=self.config_manager.classes,
                    device=self.config_manager.get("yolo.device", "cpu")
                )
                self.mqtt_worker.connection_status.connect(self.update_mqtt_status)
                self.mqtt_worker.frame_processed.connect(self.process_mqtt_result)
                self.mqtt_worker.log_message.connect(self.log_mqtt_message)
                self.mqtt_worker.start()
                self.btn_connect_mqtt.setText("断开 MQTT")
    
    def on_mqtt_server_started(self, port):
        self.btn_connect_mqtt.setText("停止 MQTT 服务端")
        self.lbl_mqtt_status.setText(f"状态: 服务端运行中 (端口: {port})")
        self.lbl_mqtt_status.setStyleSheet("background-color: #28a745; color: white;")
    
    def on_mqtt_server_stopped(self):
        self.btn_connect_mqtt.setText("启动 MQTT 服务端")
        self.lbl_mqtt_status.setText("状态: 已停止")
        self.lbl_mqtt_status.setStyleSheet("background-color: #555; color: white;")
    
    def on_mqtt_client_connected(self, client_id, port):
        self.log_mqtt_message(f"客户端 {client_id} 已连接 (端口: {port})")
    
    def on_mqtt_client_disconnected(self, client_id, port):
        self.log_mqtt_message(f"客户端 {client_id} 已断开 (端口: {port})")
    
    def on_mqtt_server_image_data(self, client_id, image_bytes):
        #self.log_mqtt_message(f"收到图像数据，字节长度: {len(image_bytes)}")
        print("收到图像数据，字节长度: ", len(image_bytes))
        if self.mqtt_inference_thread:
            self.mqtt_inference_thread.update_frame(image_bytes)

    def on_mqtt_inference_finished(self, annotated_frame, detections):
        print(f"[MainWindow] Signal received. Detections: {len(detections)}")
        self.mqtt_display.update_image(annotated_frame)
        # self.log_mqtt_message("图像已更新到显示区域") # Reduce log spam
        print("图像已更新到显示区域")
        if detections:
            self.log_result("MQTT服务端 (摄像头)", detections)
    
    def on_mqtt_server_message(self, topic, payload, client_id):
        if topic == "siot/摄像头":
            return
            
        self.log_mqtt_message(f"来自 {client_id} 的消息 - 主题: {topic}, 内容: {payload}")
        
        import base64
        import numpy as np
        
        try:
            image_data = base64.b64decode(payload)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                detections, annotated_frame, _ = self.yolo.predict(frame)
                self.mqtt_display.update_image(annotated_frame)
                if detections:
                    self.log_result(f"MQTT服务端 ({topic})", detections)
        except Exception as e:
            self.log_mqtt_message(f"处理消息时出错: {str(e)}")

    def send_manual_mqtt_message(self):
        """手动发送MQTT消息 (支持服务端和客户端模式)"""
        # Check Server
        is_server_running = self.mqtt_server and self.mqtt_server.is_running()
        # Check Client
        is_client_running = self.mqtt_worker and self.mqtt_worker.isRunning()
        
        if not is_server_running and not is_client_running:
            QMessageBox.warning(self, "错误", "MQTT 未连接 (请启动服务端或连接客户端)")
            return
            
        topic = self.edit_pub_topic.text().strip()
        message = self.edit_pub_message.text().strip()
        
        if not topic:
            QMessageBox.warning(self, "错误", "请输入 Topic")
            return
            
        if not message:
            QMessageBox.warning(self, "错误", "请输入发送内容")
            return
            
        try:
            if is_server_running:
                self.mqtt_server.publish_message(topic, message)
                self.log_mqtt_message(f"[服务端] 手动发送 - 主题: {topic}, 内容: {message}")
            elif is_client_running:
                self.mqtt_worker.publish_message(topic, message)
                # Client worker logs its own message usually, but let's double check or rely on worker's signal
                # core/mqtt_worker.py: self.log_message.emit(f"消息已发布到主题: {topic}")
                # So we might duplicate log if we log here too. But let's log specifically "Manual Send"
                self.log_mqtt_message(f"[客户端] 手动发送 - 主题: {topic}, 内容: {message}")

            self.edit_pub_message.clear() # Optional: clear message after send
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发送失败: {str(e)}")

    def update_mqtt_status(self, connected, message):
        if connected:
            self.lbl_mqtt_status.setText(f"状态: {message}")
            self.lbl_mqtt_status.setStyleSheet("background-color: #28a745; color: white;")
        else:
            self.lbl_mqtt_status.setText(f"状态: {message}")
            self.lbl_mqtt_status.setStyleSheet("background-color: #dc3545; color: white;")

    def log_mqtt_message(self, message):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.mqtt_log_counter += 1
        row = self.mqtt_log_text.rowCount()
        self.mqtt_log_text.insertRow(row)
        
        item0 = QTableWidgetItem(str(timestamp))
        item1 = QTableWidgetItem(message)
        
        self.mqtt_log_text.setItem(row, 0, item0)
        self.mqtt_log_text.setItem(row, 1, item1)
        
        if self.mqtt_log_counter > 100:
            self.mqtt_log_text.removeRow(0)
            self.mqtt_log_counter -= 1
        
        self.mqtt_log_text.scrollToBottom()

    def process_mqtt_result(self, topic, annotated_frame, detections):
        self.mqtt_display.update_image(annotated_frame)
        if detections and topic != "siot/摄像头":
            self.log_result(f"MQTT ({topic})", detections)

    # Settings
    def update_device_check_mark(self):
        if self.radio_cpu.isChecked():
            self.lbl_cpu_check.setVisible(True)
            self.lbl_gpu_check.setVisible(False)
        else:
            self.lbl_cpu_check.setVisible(False)
            self.lbl_gpu_check.setVisible(True)
    
    def save_settings(self):
        # Save MQTT Mode
        mqtt_mode = "server" if self.radio_mqtt_server.isChecked() else "client"
        self.config_manager.set("mqtt.mode", mqtt_mode)
        
        # Save MQTT Client Settings
        self.config_manager.set("mqtt.broker", self.edit_broker.text())
        self.config_manager.set("mqtt.port", self.edit_port.value())
        self.config_manager.set("mqtt.username", self.edit_user.text())
        self.config_manager.set("mqtt.password", self.edit_pass.text())
        
        # Save MQTT Server Settings
        self.config_manager.set("mqtt.server_host", self.edit_server_host.text())
        self.config_manager.set("mqtt.server_port", self.edit_server_port.value())
        
        # Save Inference Settings
        new_conf = self.spin_conf.value()
        self.config_manager.set("yolo.conf_threshold", new_conf)
        
        # Save UI Settings
        self.config_manager.set("ui.theme_color", self.combo_color.currentData())
        
        # Save Window Title
        new_title = self.edit_window_title.text().strip()
        if new_title:
             self.config_manager.set("ui.window_title", new_title)
             self.setWindowTitle(new_title)
        
        # Update local instance immediately
        if self.yolo:
            self.yolo.conf_threshold = new_conf
        
        # Save Device Settings
        new_device = "cpu" if self.radio_cpu.isChecked() else "cuda"
        old_device = self.config_manager.get("yolo.device", "cpu")
        
        # Save Model Path
        new_model_path = self.edit_model_name.text().strip()
        if not new_model_path:
            new_model_path = "best.pt" # Default
            self.edit_model_name.setText(new_model_path)
            
        old_model_path = self.config_manager.get("yolo.model_path", "best.pt")
        
        need_restart_threads = False
        
        # Check if Model or Device changed
        if new_model_path != old_model_path:
            self.config_manager.set("yolo.model_path", new_model_path)
            if self.yolo:
                self.yolo.model_path = os.path.join(os.path.dirname(self.yolo.model_path), new_model_path)
                # Re-init will happen below if device also changes, or we force it
                need_restart_threads = True
                print(f"[Settings] 模型名称已更改: {old_model_path} -> {new_model_path}")

        if new_device != old_device:
            self.config_manager.set("yolo.device", new_device)
            need_restart_threads = True
            
            # Try to switch device for local instance
            if new_device == "cuda":
                try:
                    self.yolo.set_device(new_device)
                    QMessageBox.information(self, "硬件加速", "GPU加速已启用！\n相关后台线程将自动重启以应用新设置。")
                    self.lbl_gpu_check.setVisible(True)
                    self.lbl_cpu_check.setVisible(False)
                except Exception as e:
                    self.show_gpu_error_dialog(str(e))
                    self.radio_cpu.setChecked(True)
                    self.config_manager.set("yolo.device", "cpu")
                    self.yolo.set_device("cpu")
                    self.lbl_cpu_check.setVisible(True)
                    self.lbl_gpu_check.setVisible(False)
                    new_device = "cpu"
                    need_restart_threads = False # Cancel restart if failed (or restart with cpu?)
                    # If model changed but GPU failed, we still might want to restart threads with CPU
                    if new_model_path != old_model_path:
                         need_restart_threads = True
            else:
                self.yolo.set_device(new_device)
                QMessageBox.information(self, "硬件加速", "已切换至CPU运行模式！\n相关后台线程将自动重启以应用新设置。")
                self.lbl_cpu_check.setVisible(True)
                self.lbl_gpu_check.setVisible(False)
        
        # Re-initialize local model if only model path changed (and device didn't trigger init logic above)
        if new_model_path != old_model_path and new_device == old_device:
             if self.yolo:
                 # Need to reconstruct full path logic roughly or just trust relative
                 # Inference class handles relative paths well usually if cwd is correct
                 # But YoloInference does: os.path.join(base_path, model_path)
                 self.yolo.model_path = new_model_path # YoloInference handles join internally if we pass to init, but here we are setting attr
                 # Actually self.yolo.model_path in __init__ stores the FULL path. 
                 # We need to be careful. 
                 # Let's just re-instantiate or call init_model? 
                 # The init_model uses self.model_path. 
                 # We need to update self.model_path to the full path.
                 
                 # Simpler: Just make YoloInference reload from the filename
                 # But YoloInference.__init__ calculates base_path. 
                 # We can hack it or add a method. 
                 # For now, let's just use the logic from __init__ roughly:
                 if getattr(sys, 'frozen', False):
                    base_path = sys._MEIPASS
                 else:
                    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                 self.yolo.model_path = os.path.join(base_path, new_model_path)
                 
                 self.yolo.init_model()
                 QMessageBox.information(self, "模型设置", f"模型已切换为: {new_model_path}")


        if need_restart_threads:
            # Restart MqttInferenceThread if running
            if self.mqtt_inference_thread and self.mqtt_inference_thread.isRunning():
                print(f"[Settings] 重启 MQTT 推理线程以应用新设置")
                self.mqtt_inference_thread.stop()
                self.mqtt_inference_thread.wait() # Ensure it stops
                # Re-create and start
                self.mqtt_inference_thread = MqttInferenceThread(
                    model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
                    conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
                    classes_dict=self.config_manager.classes,
                    device=self.config_manager.get("yolo.device", "cpu")
                )
                self.mqtt_inference_thread.inference_finished.connect(self.on_mqtt_inference_finished)
                self.mqtt_inference_thread.error_occurred.connect(lambda err: self.log_mqtt_message(f"推理错误: {err}"))
                self.mqtt_inference_thread.start()
                
            # Restart VideoThread if running (Local Camera)
            if self.video_thread and self.video_thread.isRunning():
                 print(f"[Settings] 重启摄像头线程以应用新设置")
                 self.on_btn_start_cam_clicked() # Stop
                 self.on_btn_start_cam_clicked() # Start (will read new config)
            
            # Restart HTTP Thread if running
            if self.http_thread and self.http_thread.isRunning():
                 print(f"[Settings] 重启HTTP监控线程以应用新设置")
                 self.on_btn_start_http_clicked() # Stop
                 self.on_btn_start_http_clicked() # Start
            
        # Save UI Settings
        new_mode = "dark" if self.combo_mode.currentIndex() == 0 else "light"
        new_color = self.combo_color.currentData()
        
        self.config_manager.set("ui.theme", new_mode)
        self.config_manager.set("ui.theme_color", new_color)
        self.apply_styles()
        
        topics = []
        for i in range(self.topic_table.rowCount()):
            name_item = self.topic_table.item(i, 0)
            topic_item = self.topic_table.item(i, 1)
            if name_item and topic_item:
                topics.append({"name": name_item.text(), "topic": topic_item.text()})
        
        self.config_manager.set("mqtt.topics", topics)
        QMessageBox.information(self, "设置", "配置保存成功！")

    def show_gpu_error_dialog(self, error_message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Warning)
        error_dialog.setWindowTitle("GPU初始化失败")
        
        error_text = f"<b>GPU初始化失败</b><br><br>"
        error_text += f"错误类型: {error_message}<br><br>"
        error_text += "建议: 系统已自动切换至CPU运行模式。<br>"
        error_text += "如需使用GPU加速，请检查：<br>"
        error_text += "1. 是否安装了CUDA和PyTorch GPU版本<br>"
        error_text += "2. GPU驱动是否正常工作<br>"
        error_text += "3. GPU内存是否充足"
        
        error_dialog.setText(error_text)
        error_dialog.setStandardButtons(QMessageBox.Ok)
        error_dialog.button(QMessageBox.Ok).setText("确定")
        error_dialog.exec()

    # --- Batch Inference Methods ---
    
    def start_batch_inference(self, image_paths):
        if self.batch_inference_thread and self.batch_inference_thread.isRunning():
            self.batch_inference_thread.stop()
            self.batch_inference_thread.wait()
        
        self.batch_results = []
        self.current_batch_index = 0
        
        device = self.config_manager.get("yolo.device", "cpu")
        if device == "cuda":
            device = "cuda"
        else:
            device = "cpu"
        
        self.batch_inference_thread = BatchInferenceThread(
            image_paths=image_paths,
            model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
            conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
            classes_dict=self.config_manager.classes,
            device=device
        )
        
        self.batch_inference_thread.progress_updated.connect(self.on_batch_progress)
        self.batch_inference_thread.result_ready.connect(self.on_batch_result)
        self.batch_inference_thread.batch_finished.connect(self.on_batch_finished)
        self.batch_inference_thread.error_occurred.connect(self.on_batch_error)
        
        self.batch_progress.setVisible(True)
        self.batch_progress_label.setVisible(True)
        self.batch_index_label.setVisible(True)
        self.batch_progress.setValue(0)
        self.batch_progress.setMaximum(len(image_paths))
        self.batch_progress_label.setText("开始处理...")
        self.batch_index_label.setText("0 / 0")
        
        self.btn_prev_result.setEnabled(False)
        self.btn_next_result.setEnabled(False)
        self.btn_stop_batch.setEnabled(True)
        
        self.batch_inference_thread.start()
    
    def on_batch_progress(self, current, total, message):
        self.batch_progress.setValue(current)
        self.batch_progress_label.setText(f"{message} ({current}/{total})")
    
    def on_batch_result(self, filename, original_image, annotated_image, detections):
        result = {
            'filename': filename,
            'original_image': original_image,
            'annotated_image': annotated_image,
            'detections': detections
        }
        self.batch_results.append(result)
        
        if len(self.batch_results) == 1:
            self.current_batch_index = 0
            self.show_batch_result(0)
            self.update_batch_navigation()
        
        self.batch_index_label.setText(f"{len(self.batch_results)} / {self.batch_progress.maximum()}")
    
    def on_batch_finished(self, count):
        self.batch_progress_label.setText(f"处理完成！共处理 {count} 张图片")
        self.btn_stop_batch.setEnabled(False)
        
        if count > 0:
            self.update_batch_navigation()
    
    def on_batch_error(self, error_message):
        QMessageBox.warning(self, "批量推理错误", error_message)
    
    def stop_batch_inference(self):
        if self.batch_inference_thread and self.batch_inference_thread.isRunning():
            self.batch_inference_thread.stop()
            self.batch_progress_label.setText("处理已停止")
            self.btn_stop_batch.setEnabled(False)
    
    def show_batch_result(self, index):
        if 0 <= index < len(self.batch_results):
            result = self.batch_results[index]
            self.local_display_orig.update_image(result['original_image'])
            self.local_display_res.update_image(result['annotated_image'])
            
            if result['detections']:
                self.log_result(f"批量推理[{index+1}]", result['detections'])
    
    def show_prev_batch_result(self):
        if self.current_batch_index > 0:
            self.current_batch_index -= 1
            self.show_batch_result(self.current_batch_index)
            self.update_batch_navigation()
    
    def show_next_batch_result(self):
        if self.current_batch_index < len(self.batch_results) - 1:
            self.current_batch_index += 1
            self.show_batch_result(self.current_batch_index)
            self.update_batch_navigation()
    
    def update_batch_navigation(self):
        total = len(self.batch_results)
        if total > 0:
            self.batch_index_label.setText(f"{self.current_batch_index + 1} / {total}")
            self.btn_prev_result.setEnabled(self.current_batch_index > 0)
            self.btn_next_result.setEnabled(self.current_batch_index < total - 1)
        else:
            self.batch_index_label.setText("0 / 0")
            self.btn_prev_result.setEnabled(False)
            self.btn_next_result.setEnabled(False)

    # --- MQTT Helper Methods ---
    
    def on_mqtt_mode_changed(self):
        if self.radio_mqtt_server.isChecked():
            self.client_settings_widget.setVisible(False)
            self.server_settings_widget.setVisible(True)
            self.config_manager.set("mqtt.mode", "server")
            if hasattr(self, 'btn_connect_mqtt'):
                self.btn_connect_mqtt.setText("启动 MQTT 服务端")
                self.lbl_mqtt_status.setText("状态: 未启动")
                self.lbl_mqtt_status.setStyleSheet("background-color: #555; color: white;")
        else:
            self.client_settings_widget.setVisible(True)
            self.server_settings_widget.setVisible(False)
            self.config_manager.set("mqtt.mode", "client")
            if hasattr(self, 'btn_connect_mqtt'):
                self.btn_connect_mqtt.setText("连接 MQTT")
                self.lbl_mqtt_status.setText("状态: 未连接")
                self.lbl_mqtt_status.setStyleSheet("background-color: #555; color: white;")
    
    def on_mqtt_server_changed(self, index):
        server_type = self.combo_mqtt_server.currentData()
        
        if server_type == "local":
            self.edit_broker.setText("127.0.0.1")
            self.edit_port.setValue(1883)
            self.edit_user.setText("")
            self.edit_pass.setText("")
        elif server_type == "remote":
            self.edit_broker.setText("10.1.2.3")
            self.edit_port.setValue(1883)
            self.edit_user.setText("siot")
            self.edit_pass.setText("dfrobot")
    
    def test_mqtt_connection(self):
        broker = self.edit_broker.text().strip()
        port = self.edit_port.value()
        username = self.edit_user.text().strip()
        password = self.edit_pass.text().strip()
        
        if not broker:
            QMessageBox.warning(self, "错误", "请输入MQTT服务器地址")
            return
        
        self.btn_test_mqtt.setEnabled(False)
        self.btn_test_mqtt.setText("测试中...")
        
        import paho.mqtt.client as mqtt
        test_client = mqtt.Client()
        
        if username and password:
            test_client.username_pw_set(username, password)
        
        def on_connect(client, userdata, flags, rc):
            client.disconnect()
        
        test_client.on_connect = on_connect
        
        try:
            test_client.connect(broker, port, 5)
            test_client.loop_start()
            
            import time
            timeout = 5
            start_time = time.time()
            
            while not test_client.is_connected() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            test_client.loop_stop()
            
            if test_client.is_connected():
                QMessageBox.information(self, "测试成功", f"成功连接到MQTT服务器\n\n服务器: {broker}\n端口: {port}")
            else:
                QMessageBox.warning(self, "测试失败", f"无法连接到MQTT服务器\n\n服务器: {broker}\n端口: {port}\n\n请检查:\n1. 服务器地址是否正确\n2. 网络连接是否正常\n3. 服务器是否正在运行")
                
        except Exception as e:
            QMessageBox.critical(self, "测试错误", f"连接测试时发生错误:\n\n{str(e)}")
        finally:
            self.btn_test_mqtt.setEnabled(True)
            self.btn_test_mqtt.setText("测试连接")

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if self.http_thread:
            self.http_thread.stop()
        if self.mqtt_worker:
            self.mqtt_worker.stop()
        if self.mqtt_inference_thread:
            self.mqtt_inference_thread.stop()
        if self.batch_inference_thread:
            self.batch_inference_thread.stop()
        event.accept()
