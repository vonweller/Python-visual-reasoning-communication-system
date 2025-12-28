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
from core.video_thread import VideoThread
from core.batch_inference_thread import BatchInferenceThread
from ui.widgets import ImageDisplayWidget, LogTableWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv12的氮磷钾农作物需求识别系统")
        self.resize(1200, 800)

        # Load Config
        self.config_manager = ConfigManager()
        self.yolo = YoloInference(
            model_path=self.config_manager.get("yolo.model_path", "yolov8n.pt"),
            conf_threshold=self.config_manager.get("yolo.conf_threshold", 0.5),
            classes_dict=self.config_manager.classes
        )

        # Workers
        self.mqtt_worker = None
        self.video_thread = None
        self.http_thread = None
        self.batch_inference_thread = None

        # Batch inference data
        self.batch_results = []
        self.current_batch_index = 0

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
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
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
        self.btn_connect_mqtt = QPushButton("连接 MQTT")
        self.btn_connect_mqtt.clicked.connect(self.toggle_mqtt)
        self.lbl_mqtt_status = QPushButton("状态: 未连接")
        self.lbl_mqtt_status.setEnabled(False)
        self.lbl_mqtt_status.setStyleSheet("background-color: #555; color: white;")
        
        controls_layout.addWidget(self.btn_connect_mqtt)
        controls_layout.addWidget(self.lbl_mqtt_status)
        controls_layout.addStretch()
        
        self.mqtt_display = ImageDisplayWidget("MQTT 画面")
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.mqtt_display)

    def setup_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        
        # Connection Settings
        form_layout = QFormLayout()
        self.edit_broker = QLineEdit(self.config_manager.get("mqtt.broker"))
        self.edit_port = QSpinBox()
        self.edit_port.setRange(1, 65535)
        self.edit_port.setValue(self.config_manager.get("mqtt.port"))
        
        self.edit_user = QLineEdit(self.config_manager.get("mqtt.username", ""))
        self.edit_pass = QLineEdit(self.config_manager.get("mqtt.password", ""))
        self.edit_pass.setEchoMode(QLineEdit.Password)
        
        form_layout.addRow("MQTT 服务器:", self.edit_broker)
        form_layout.addRow("MQTT 端口:", self.edit_port)
        form_layout.addRow("MQTT 用户名:", self.edit_user)
        form_layout.addRow("MQTT 密码:", self.edit_pass)
        
        layout.addLayout(form_layout)
        
        # Inference Settings
        inf_group = QGroupBox("推理设置")
        inf_layout = QFormLayout(inf_group)
        
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(self.config_manager.get("yolo.conf_threshold", 0.5))
        
        inf_layout.addRow("置信度阈值:", self.spin_conf)
        
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
                classes_dict=self.config_manager.classes
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
                classes_dict=self.config_manager.classes
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
                classes_dict=self.config_manager.classes
            )
            self.mqtt_worker.connection_status.connect(self.update_mqtt_status)
            self.mqtt_worker.frame_processed.connect(self.process_mqtt_result)
            self.mqtt_worker.log_message.connect(self.log_mqtt_message)
            self.mqtt_worker.start()
            self.btn_connect_mqtt.setText("断开 MQTT")

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
        self.log_table.add_record(timestamp, "MQTT日志", message, "", "")

    def process_mqtt_result(self, topic, annotated_frame, detections):
        self.mqtt_display.update_image(annotated_frame)
        if detections:
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
        self.config_manager.set("mqtt.broker", self.edit_broker.text())
        self.config_manager.set("mqtt.port", self.edit_port.value())
        self.config_manager.set("mqtt.username", self.edit_user.text())
        self.config_manager.set("mqtt.password", self.edit_pass.text())
        
        # Save Inference Settings
        new_conf = self.spin_conf.value()
        self.config_manager.set("yolo.conf_threshold", new_conf)
        
        # Update local instance immediately
        if self.yolo:
            self.yolo.conf_threshold = new_conf
        
        # Save Device Settings
        new_device = "cpu" if self.radio_cpu.isChecked() else "cuda"
        old_device = self.config_manager.get("yolo.device", "cpu")
        
        if new_device != old_device:
            self.config_manager.set("yolo.device", new_device)
            
            # Try to switch device
            if new_device == "cuda":
                try:
                    self.yolo.set_device(new_device)
                    QMessageBox.information(self, "硬件加速", "GPU加速已启用！")
                    self.lbl_gpu_check.setVisible(True)
                    self.lbl_cpu_check.setVisible(False)
                except Exception as e:
                    # GPU initialization failed, show error dialog and switch back to CPU
                    self.show_gpu_error_dialog(str(e))
                    self.radio_cpu.setChecked(True)
                    self.config_manager.set("yolo.device", "cpu")
                    self.yolo.set_device("cpu")
                    self.lbl_cpu_check.setVisible(True)
                    self.lbl_gpu_check.setVisible(False)
            else:
                self.yolo.set_device(new_device)
                QMessageBox.information(self, "硬件加速", "已切换至CPU运行模式！")
                self.lbl_cpu_check.setVisible(True)
                self.lbl_gpu_check.setVisible(False)
            
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

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if self.http_thread:
            self.http_thread.stop()
        if self.mqtt_worker:
            self.mqtt_worker.stop()
        if self.batch_inference_thread:
            self.batch_inference_thread.stop()
        event.accept()
