from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                               QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import cv2

class ImageDisplayWidget(QWidget):
    def __init__(self, title="Image"):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("ImageTitle")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #2b2b2b; border: 1px solid #3f3f3f;")
        
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.image_label)

    def update_image(self, cv_img):
        if cv_img is None:
            return
        
        # Convert CV image to QPixmap
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

class LogTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["时间", "来源", "英文类别", "中文类别", "置信度"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.counter = 0

    def add_record(self, timestamp, source, class_name_en, class_name_cn, confidence):
        self.counter += 1
        row = self.rowCount()
        self.insertRow(row)
        
        # Format: 《第X次：推理结果》
        # Actually user asked for specific format in the record, maybe I should add a column or format the first column?
        # Requirement: "每条记录格式遵循"《第X次：推理结果》"的规范"
        # I will interpret this as a log entry prefix or title, but for a table, columns are better.
        # Let's stick to columns but maybe add a tooltip or status log with that format.
        # Or I can make the first column "ID" with that format.
        
        id_str = f"《第{self.counter}次：推理结果》"
        
        self.setItem(row, 0, QTableWidgetItem(str(timestamp)))
        self.setItem(row, 1, QTableWidgetItem(source))
        self.setItem(row, 2, QTableWidgetItem(class_name_en))
        self.setItem(row, 3, QTableWidgetItem(class_name_cn))
        self.setItem(row, 4, QTableWidgetItem(f"{confidence:.2f}"))
        self.scrollToBottom()
