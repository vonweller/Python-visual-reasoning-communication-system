# 项目解读文档 - 基于YOLOv12的氮磷钾农作物需求识别系统

> 本文档旨在为AI助手和开发者提供项目的完整技术解读，便于理解系统架构和进行二次开发。

## 1. 项目概述

### 1.1 功能简介
本系统是一个基于 **YOLOv8/YOLOv12** 的农作物病害识别系统，支持多种图像输入方式，并可通过MQTT协议与远程设备通信。

### 1.2 主要功能模块
| 模块 | 功能描述 |
|------|----------|
| 本地图片 | 加载本地图片进行推理，支持单张和批量处理 |
| 摄像头 | 实时捕获本地摄像头画面进行推理 |
| HTTP监控 | 连接HTTP视频流（如ESP32-CAM）进行远程监控推理 |
| MQTT远程 | 作为MQTT服务端/客户端接收远程设备发送的图像进行推理 |
| 设置 | 配置MQTT连接、推理参数、硬件加速、界面主题等 |

---

## 2. 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.x | 主开发语言 |
| PySide6 (Qt6) | GUI框架 |
| Ultralytics YOLO | 目标检测模型 |
| OpenCV (cv2) | 图像处理 |
| Paho-MQTT | MQTT客户端协议 |
| 自定义MQTT服务端 | 实现MQTT Broker功能 |
| Pillow (PIL) | 中文标签绘制 |

---

## 3. 项目结构

```
Python-visual-reasoning-communication-system/
├── main.py                    # 程序入口
├── requirements.txt           # Python依赖
├── best.pt                    # 训练好的YOLO模型（农作物病害）
├── yolov8n.pt                 # 预训练YOLO模型（备用）
├── build.bat                  # 打包脚本
├── build_exe.spec             # PyInstaller配置
│
├── config/                    # 配置文件目录
│   ├── config.json            # 主配置文件
│   └── classes.json           # 类别中英文映射
│
├── core/                      # 核心业务逻辑
│   ├── inference.py           # YOLO推理引擎
│   ├── mqtt_server.py         # MQTT服务端（自定义协议实现）
│   ├── mqtt_worker.py         # MQTT客户端工作线程
│   ├── video_thread.py        # 摄像头/HTTP视频流线程
│   ├── batch_inference_thread.py  # 批量推理线程
│   └── config_manager.py      # 配置管理器
│
├── ui/                        # 用户界面
│   ├── main_window.py         # 主窗口（所有UI逻辑）
│   ├── widgets.py             # 自定义控件
│   ├── styles.qss             # 深色主题样式
│   └── styles_light.qss       # 浅色主题样式
│
└── dist/                      # 打包输出目录
```

---

## 4. 核心模块详解

### 4.1 推理引擎 (`core/inference.py`)

**类**: `YoloInference`

**职责**: 加载YOLO模型并执行推理，支持CPU/GPU切换

**关键方法**:
```python
def __init__(self, model_path, conf_threshold, classes_dict, device="cpu")
    # model_path: 模型文件路径
    # conf_threshold: 置信度阈值
    # classes_dict: 类别ID到中文名的映射
    # device: "cpu" 或 "cuda"

def predict(self, image) -> (detections, annotated_frame, inference_time)
    # image: OpenCV格式的图像 (BGR numpy array)
    # 返回: 检测结果列表, 标注后的图像, 推理耗时(ms)

def set_device(self, device)
    # 动态切换推理设备
```

**检测结果格式**:
```python
{
    "class_id": int,           # 类别ID
    "class_name_en": str,      # 英文类名（模型原始）
    "class_name_cn": str,      # 中文类名（来自classes.json）
    "confidence": float,       # 置信度
    "bbox": [x1, y1, x2, y2]   # 边界框坐标
}
```

---

### 4.2 MQTT服务端 (`core/mqtt_server.py`)

**类**: `MqttServer(QThread)`

**职责**: 作为MQTT Broker，接收远程客户端（如siot）发送的图像数据

**协议实现**: 实现了真正的MQTT协议解析，支持以下包类型：
| 包类型 | 代码 | 功能 |
|--------|------|------|
| CONNECT | 1 | 处理客户端连接，返回CONNACK |
| PUBLISH | 3 | 解析主题和载荷，处理图像数据 |
| SUBSCRIBE | 8 | 处理订阅请求，返回SUBACK |
| UNSUBSCRIBE | 10 | 处理取消订阅 |
| PINGREQ | 12 | 心跳响应 |
| DISCONNECT | 14 | 客户端断开 |

**关键信号**:
```python
image_data_received = Signal(str, bytes)  # 收到图像数据 (client_id, image_bytes)
message_received = Signal(str, str, str)  # 收到普通消息 (topic, payload, client_id)
client_connected = Signal(str, int)       # 客户端连接
client_disconnected = Signal(str, int)    # 客户端断开
```

**图像数据格式**: 客户端发送的是BASE64编码的图像，格式为：
```
data:image/png;base64,/9j/4AAQSkZJRg...
```

---

### 4.3 MQTT客户端 (`core/mqtt_worker.py`)

**类**: `MqttWorker(QThread)`

**职责**: 作为MQTT客户端连接到外部Broker，订阅主题并处理图像

**关键方法**:
```python
def on_message(self, client, userdata, msg)
    # 自动解析BASE64图像并进行YOLO推理

def publish_message(self, topic, payload)
    # 发布消息到指定主题
```

---

### 4.4 视频线程 (`core/video_thread.py`)

**类**: `VideoThread(QThread)`

**职责**: 在独立线程中捕获视频帧并推理

**支持的视频源**:
- 本地摄像头: `camera_id=0`
- HTTP流: `camera_id="http://192.168.x.x:81/stream"`

---

### 4.5 主窗口 (`ui/main_window.py`)

**类**: `MainWindow(QMainWindow)`

**职责**: 所有UI逻辑和用户交互的中心

**关键成员变量**:
```python
self.yolo          # YoloInference实例（主窗口推理用）
self.config_manager # ConfigManager实例
self.mqtt_server   # MqttServer实例（服务端模式）
self.mqtt_worker   # MqttWorker实例（客户端模式）
self.video_thread  # VideoThread实例（摄像头）
self.http_thread   # VideoThread实例（HTTP流）
```

**MQTT帧处理优化**:
```python
self.mqtt_frame_interval = 0.1  # 最小处理间隔100ms
self.latest_mqtt_frame = None   # 保存最新帧，丢弃旧帧
```

---

## 5. 配置文件说明

### 5.1 config.json
```json
{
    "mqtt": {
        "mode": "server",           // "server" 或 "client"
        "broker": "10.1.2.3",       // 客户端模式的Broker地址
        "port": 1883,
        "username": "siot",
        "password": "dfrobot",
        "server_host": "0.0.0.0",   // 服务端监听地址
        "server_port": 1883,
        "publish_topic": "siot/推理结果",  // 推理结果发布主题
        "topics": [...]             // 订阅的主题列表
    },
    "yolo": {
        "model_path": "best.pt",    // 模型文件
        "conf_threshold": 0.88,     // 置信度阈值
        "device": "cpu"             // "cpu" 或 "cuda"
    },
    "ui": {
        "theme": "light",           // "dark" 或 "light"
        "theme_color": "#28a745"    // 主题色
    }
}
```

### 5.2 classes.json
```json
{
    "0": "角斑病",
    "1": "叶斑病",
    "2": "白粉病叶",
    ...
}
```

---

## 6. 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                         远程设备 (siot客户端)                      │
│  ┌─────────┐    BASE64图像     ┌─────────────┐                  │
│  │ 摄像头   │ ───────────────▶ │ MQTT PUBLISH │                  │
│  └─────────┘                   └─────────────┘                  │
└──────────────────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      本地服务端 (Python程序)                       │
│                                                                  │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│  │ MqttServer  │─────▶│ MainWindow  │─────▶│ ImageDisplay│      │
│  │ (协议解析)   │ 信号  │ (帧处理)    │ 更新  │ (UI显示)    │      │
│  └─────────────┘      └──────┬──────┘      └─────────────┘      │
│                              │                                   │
│                              ▼                                   │
│                       ┌─────────────┐      ┌─────────────┐      │
│                       │YoloInference│─────▶│  LogTable   │      │
│                       │ (模型推理)   │ 结果  │ (日志记录)   │      │
│                       └─────────────┘      └─────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. 二次开发指南

### 7.1 添加新的图像输入源
1. 创建新的线程类继承 `QThread`
2. 发射 `frame_processed` 信号，参数为 `(annotated_frame, detections)`
3. 在 `MainWindow` 中添加对应的Tab和控制逻辑

### 7.2 添加新的检测类别
1. 训练新的YOLO模型
2. 更新 `config/classes.json` 添加类别映射
3. 替换 `best.pt` 模型文件

### 7.3 修改MQTT通信协议
- 服务端协议实现: `core/mqtt_server.py`
- 客户端逻辑: `core/mqtt_worker.py`
- 主题配置: `config/config.json` 的 `mqtt.topics`

### 7.4 自定义UI样式
- 深色主题: `ui/styles.qss`
- 浅色主题: `ui/styles_light.qss`
- 主题色通过配置中的 `ui.theme_color` 动态替换

---

## 8. 常见问题

### Q: MQTT服务端收不到客户端消息？
**A**: 检查客户端是否使用标准MQTT协议（如paho-mqtt/siot），本服务端实现了真正的MQTT协议解析。

### Q: 画面延迟太高？
**A**: 调整 `main_window.py` 中的 `mqtt_frame_interval` 值，减小可降低延迟但增加CPU负载。

### Q: GPU推理不生效？
**A**: 
1. 确认安装了CUDA版本的PyTorch
2. 检查 `config.json` 中 `yolo.device` 设置为 `"cuda"`
3. 查看启动日志确认: `[YoloInference] 模型已加载，使用设备: cuda`

### Q: 程序打包？
**A**: 运行 `build.bat`，使用PyInstaller打包，输出到 `dist/` 目录。

---

## 9. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | - | 初始版本 |
| 1.1 | 2025-12-30 | 修复MQTT协议兼容性，优化帧处理延迟，修复设备配置传递 |
