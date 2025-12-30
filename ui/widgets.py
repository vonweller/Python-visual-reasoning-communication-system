from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from PySide6.QtCore import Qt, QSize
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
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setWordWrap(True)
        self.counter = 0

    def add_record(self, timestamp, source, class_name_en, class_name_cn, confidence):
        self.counter += 1
        row = self.rowCount()
        self.insertRow(row)
        
        item0 = QTableWidgetItem(str(timestamp))
        item1 = QTableWidgetItem(source)
        item2 = QTableWidgetItem(class_name_en)
        item3 = QTableWidgetItem(class_name_cn)
        
        try:
            confidence_float = float(confidence)
            item4 = QTableWidgetItem(f"{confidence_float:.2f}")
        except (ValueError, TypeError):
            item4 = QTableWidgetItem(str(confidence))
        
        self.setItem(row, 0, item0)
        self.setItem(row, 1, item1)
        self.setItem(row, 2, item2)
        self.setItem(row, 3, item3)
        self.setItem(row, 4, item4)
        
        self.resizeRowToContents(row)
        self.scrollToBottom()
