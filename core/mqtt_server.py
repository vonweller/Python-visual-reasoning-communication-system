import socket
import threading
import time
from PySide6.QtCore import QThread, Signal, QMutex
import json
import struct
from collections import deque
import base64

class MqttServer(QThread):
    client_connected = Signal(str, int)
    client_disconnected = Signal(str, int)
    message_received = Signal(str, str, str)
    image_data_received = Signal(str, bytes)
    server_started = Signal(int)
    server_stopped = Signal()
    log_message = Signal(str)

    def __init__(self, host="0.0.0.0", port=1883):
        super().__init__()
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.clients = {}
        self.client_counter = 0
        self.subscriptions = {}
        self.topics = {}
        self.mutex = QMutex()
        self.message_queue = deque(maxlen=1000)
        self.log_queue = deque(maxlen=1000)
        self.last_log_time = time.time()
        self.log_interval = 0.1

    def get_local_ip(self):
        try:
            ips = []
            
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip.startswith("192.168."):
                return local_ip
            
            try:
                for interface in socket.getaddrinfo(hostname, None):
                    ip = interface[4][0]
                    if ip.startswith("192.168.") and ip not in ips:
                        ips.append(ip)
            except:
                pass
            
            if ips:
                return ips[0]
            
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            if local_ip.startswith("192.168."):
                return local_ip
            
            return local_ip
        except:
            return "127.0.0.1"

    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            self.server_started.emit(self.port)
            
            local_ip = self.get_local_ip()
            self.log_message.emit(f"MQTT服务端已启动")
            self.log_message.emit(f"监听地址: {self.host}:{self.port}")
            self.log_message.emit(f"客户端请连接: {local_ip}:{self.port}")
            
            while self.running:
                try:
                    self.server_socket.settimeout(0.1)
                    client_socket, address = self.server_socket.accept()
                    
                    self.client_counter += 1
                    client_id = f"client_{self.client_counter}"
                    
                    self.mutex.lock()
                    self.clients[client_id] = {
                        'socket': client_socket,
                        'address': address,
                        'connected': True
                    }
                    self.mutex.unlock()
                    
                    self.client_connected.emit(client_id, address[1])
                    self.log_message.emit(f"客户端已连接: {client_id} ({address[0]}:{address[1]})")
                    
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_id, client_socket)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    self.process_queues()
                    continue
                except Exception as e:
                    if self.running:
                        self.log_message.emit(f"接受连接时出错: {str(e)}")
                        
        except Exception as e:
            self.log_message.emit(f"MQTT服务端启动失败: {str(e)}")
        finally:
            self.stop()

    def process_queues(self):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.mutex.lock()
            while self.log_queue:
                log_msg = self.log_queue.popleft()
                self.log_message.emit(log_msg)
            self.mutex.unlock()
            self.last_log_time = current_time

    def safe_log(self, message):
        print(f"[MQTT服务端] {message}")
        self.mutex.lock()
        self.log_queue.append(message)
        self.mutex.unlock()

    def decode_remaining_length(self, data, start_pos):
        """解码MQTT剩余长度字段（变长编码）"""
        multiplier = 1
        value = 0
        pos = start_pos
        while pos < len(data):
            byte = data[pos]
            value += (byte & 127) * multiplier
            multiplier *= 128
            pos += 1
            if (byte & 128) == 0:
                break
            if multiplier > 128 * 128 * 128:
                raise ValueError("剩余长度编码错误")
        return value, pos

    def encode_remaining_length(self, length):
        """编码MQTT剩余长度字段"""
        encoded = bytearray()
        while True:
            byte = length % 128
            length = length // 128
            if length > 0:
                byte = byte | 128
            encoded.append(byte)
            if length == 0:
                break
        return bytes(encoded)

    def decode_string(self, data, pos):
        """解码MQTT UTF-8字符串"""
        if pos + 2 > len(data):
            return None, pos
        str_len = struct.unpack(">H", data[pos:pos+2])[0]
        pos += 2
        if pos + str_len > len(data):
            return None, pos
        string = data[pos:pos+str_len].decode('utf-8')
        pos += str_len
        return string, pos

    def handle_client(self, client_id, client_socket):
        buffer = b''
        try:
            while self.running and self.clients.get(client_id, {}).get('connected', False):
                try:
                    client_socket.settimeout(0.1)
                    data = client_socket.recv(65536)
                    
                    if not data:
                        break
                    
                    buffer += data
                    
                    # 处理缓冲区中的所有完整MQTT包
                    while len(buffer) >= 2:
                        # 解析固定头
                        packet_type = (buffer[0] >> 4) & 0x0F
                        
                        # 解码剩余长度
                        try:
                            remaining_length, header_end = self.decode_remaining_length(buffer, 1)
                        except ValueError:
                            self.safe_log("剩余长度解码错误，清空缓冲区")
                            buffer = b''
                            break
                        
                        # 计算完整包长度
                        total_length = header_end + remaining_length
                        
                        # 检查是否收到完整的包
                        if len(buffer) < total_length:
                            break
                        
                        # 提取完整的包
                        packet = buffer[:total_length]
                        buffer = buffer[total_length:]
                        
                        # 处理不同类型的MQTT包
                        if packet_type == 1:  # CONNECT
                            self.handle_connect(client_id, client_socket, packet)
                        elif packet_type == 3:  # PUBLISH
                            self.handle_publish(client_id, packet)
                        elif packet_type == 8:  # SUBSCRIBE
                            self.handle_subscribe(client_id, client_socket, packet)
                        elif packet_type == 10:  # UNSUBSCRIBE
                            self.handle_unsubscribe(client_id, client_socket, packet)
                        elif packet_type == 12:  # PINGREQ
                            self.handle_pingreq(client_id, client_socket)
                        elif packet_type == 14:  # DISCONNECT
                            self.safe_log(f"客户端 {client_id} 发送DISCONNECT")
                            break
                        else:
                            self.safe_log(f"收到未知包类型: {packet_type}")
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.clients.get(client_id, {}).get('connected', False):
                        self.safe_log(f"处理客户端 {client_id} 数据时出错: {str(e)}")
                        import traceback
                        self.safe_log(f"详细错误: {traceback.format_exc()}")
                    break
                    
        except Exception as e:
            self.safe_log(f"客户端 {client_id} 处理线程异常: {str(e)}")
        finally:
            self.disconnect_client(client_id)

    def handle_connect(self, client_id, client_socket, packet):
        """处理CONNECT包并发送CONNACK"""
        self.safe_log(f"客户端 {client_id} 发送CONNECT包")
        
        # 发送CONNACK: 0x20 0x02 0x00 0x00 (连接接受)
        connack = bytes([0x20, 0x02, 0x00, 0x00])
        try:
            client_socket.sendall(connack)
            self.safe_log(f"向客户端 {client_id} 发送CONNACK")
        except Exception as e:
            self.safe_log(f"发送CONNACK失败: {str(e)}")

    def handle_subscribe(self, client_id, client_socket, packet):
        """处理SUBSCRIBE包并发送SUBACK"""
        try:
            # 解析固定头
            remaining_length, header_end = self.decode_remaining_length(packet, 1)
            
            # 解析可变头（包标识符）
            packet_id = struct.unpack(">H", packet[header_end:header_end+2])[0]
            pos = header_end + 2
            
            # 解析主题过滤器
            topics = []
            return_codes = []
            
            while pos < len(packet):
                topic, pos = self.decode_string(packet, pos)
                if topic is None:
                    break
                qos = packet[pos] if pos < len(packet) else 0
                pos += 1
                topics.append(topic)
                return_codes.append(min(qos, 2))  # 返回授予的QoS
                
                self.subscribe_topic(client_id, topic)
                self.safe_log(f"客户端 {client_id} 订阅主题: {topic} (QoS: {qos})")
            
            # 发送SUBACK
            suback = bytearray([0x90])  # SUBACK固定头
            suback_payload = struct.pack(">H", packet_id) + bytes(return_codes)
            suback += self.encode_remaining_length(len(suback_payload))
            suback += suback_payload
            
            client_socket.sendall(bytes(suback))
            self.safe_log(f"向客户端 {client_id} 发送SUBACK")
            
        except Exception as e:
            self.safe_log(f"处理SUBSCRIBE包出错: {str(e)}")
            import traceback
            self.safe_log(f"详细错误: {traceback.format_exc()}")

    def handle_unsubscribe(self, client_id, client_socket, packet):
        """处理UNSUBSCRIBE包并发送UNSUBACK"""
        try:
            remaining_length, header_end = self.decode_remaining_length(packet, 1)
            packet_id = struct.unpack(">H", packet[header_end:header_end+2])[0]
            pos = header_end + 2
            
            while pos < len(packet):
                topic, pos = self.decode_string(packet, pos)
                if topic is None:
                    break
                self.unsubscribe_topic(client_id, topic)
            
            # 发送UNSUBACK
            unsuback = bytes([0xB0, 0x02]) + struct.pack(">H", packet_id)
            client_socket.sendall(unsuback)
            
        except Exception as e:
            self.safe_log(f"处理UNSUBSCRIBE包出错: {str(e)}")

    def handle_pingreq(self, client_id, client_socket):
        """处理PINGREQ包并发送PINGRESP"""
        try:
            # 发送PINGRESP: 0xD0 0x00
            pingresp = bytes([0xD0, 0x00])
            client_socket.sendall(pingresp)
        except Exception as e:
            self.safe_log(f"发送PINGRESP失败: {str(e)}")

    def handle_publish(self, client_id, packet):
        """处理PUBLISH包"""
        try:
            # 解析固定头
            flags = packet[0] & 0x0F
            qos = (flags >> 1) & 0x03
            retain = flags & 0x01
            dup = (flags >> 3) & 0x01
            
            remaining_length, header_end = self.decode_remaining_length(packet, 1)
            pos = header_end
            
            # 解析主题名
            topic, pos = self.decode_string(packet, pos)
            if topic is None:
                self.safe_log("无法解析主题名")
                return
            
            # 如果QoS > 0，解析包标识符
            if qos > 0:
                packet_id = struct.unpack(">H", packet[pos:pos+2])[0]
                pos += 2
            
            # 剩余的是载荷
            payload = packet[pos:]
            
            # 尝试解码为UTF-8字符串
            try:
                payload_str = payload.decode('utf-8')
            except:
                payload_str = None
            
            if topic != "siot/摄像头":
                self.safe_log(f"收到PUBLISH - 主题: {topic}, 载荷长度: {len(payload)}")
            
            # 处理摄像头主题
            if topic == "siot/摄像头" and payload_str:
                self.process_camera_image(client_id, topic, payload_str)
            elif payload_str:
                # 发送消息到UI
                self.message_received.emit(topic, payload_str, client_id)
                
                # 转发给订阅者
                self.forward_to_subscribers(topic, payload)
            
        except Exception as e:
            self.safe_log(f"处理PUBLISH包出错: {str(e)}")
            import traceback
            self.safe_log(f"详细错误: {traceback.format_exc()}")

    def process_camera_image(self, client_id, topic, payload_str):
        """处理摄像头图像数据"""
        try:
            # 去掉data:image/xxx;base64,前缀
            if 'base64,' in payload_str:
                base64_data = payload_str.split('base64,', 1)[1]
            else:
                base64_data = payload_str
            
            # 清理base64数据
            base64_data = base64_data.strip()
            
            # 修复padding
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)
            
            # 解码
            image_bytes = base64.b64decode(base64_data)
            #self.safe_log(f"成功解码BASE64图像数据，字节长度: {len(image_bytes)}")
            print("成功解码BASE64图像数据，字节长度: ", len(image_bytes))
            # 发送图像数据信号
            self.image_data_received.emit(client_id, image_bytes)
            
        except Exception as e:
            self.safe_log(f"处理摄像头图像数据失败: {str(e)}")
            import traceback
            self.safe_log(f"详细错误: {traceback.format_exc()}")

    def forward_to_subscribers(self, topic, payload):
        """转发消息给订阅该主题的客户端"""
        self.mutex.lock()
        subscriptions_copy = dict(self.subscriptions)
        clients_copy = dict(self.clients)
        self.mutex.unlock()
        
        for sub_client_id, subscribed_topics in subscriptions_copy.items():
            if topic in subscribed_topics and sub_client_id in clients_copy:
                if clients_copy[sub_client_id]['connected']:
                    try:
                        # 构建PUBLISH包
                        publish_packet = self.build_publish_packet(topic, payload)
                        clients_copy[sub_client_id]['socket'].sendall(publish_packet)
                    except Exception as e:
                        self.safe_log(f"转发消息到 {sub_client_id} 失败: {str(e)}")

    def build_publish_packet(self, topic, payload):
        """构建MQTT PUBLISH包"""
        # 编码主题
        topic_bytes = topic.encode('utf-8')
        topic_length = struct.pack(">H", len(topic_bytes))
        
        # 可变头 = 主题长度 + 主题
        variable_header = topic_length + topic_bytes
        
        # 计算剩余长度
        remaining = variable_header + payload
        
        # 固定头
        fixed_header = bytes([0x30]) + self.encode_remaining_length(len(remaining))
        
        return fixed_header + remaining

    def disconnect_client(self, client_id):
        if client_id in self.clients:
            client_info = self.clients[client_id]
            client_info['connected'] = False
            
            try:
                client_info['socket'].close()
            except:
                pass
            
            address = client_info['address']
            self.client_disconnected.emit(client_id, address[1])
            self.safe_log(f"客户端已断开: {client_id} ({address[0]}:{address[1]})")
            
            self.mutex.lock()
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
            self.mutex.unlock()

    def broadcast_message(self, message):
        disconnected_clients = []
        
        self.mutex.lock()
        clients_copy = dict(self.clients)
        self.mutex.unlock()
        
        for client_id, client_info in clients_copy.items():
            if client_info['connected']:
                try:
                    client_info['socket'].sendall(message.encode('utf-8'))
                except Exception as e:
                    self.safe_log(f"发送消息到客户端 {client_id} 失败: {str(e)}")
                    disconnected_clients.append(client_id)
        
        for client_id in disconnected_clients:
            self.disconnect_client(client_id)

    def send_message_to_client(self, client_id, message):
        self.mutex.lock()
        client_info = self.clients.get(client_id)
        if client_info and client_info['connected']:
            self.mutex.unlock()
            try:
                client_info['socket'].sendall(message.encode('utf-8'))
                return True
            except Exception as e:
                self.safe_log(f"发送消息到客户端 {client_id} 失败: {str(e)}")
                self.disconnect_client(client_id)
                return False
        else:
            self.mutex.unlock()
            return False

    def get_connected_clients(self):
        self.mutex.lock()
        clients_list = [
            {
                'id': client_id,
                'address': f"{info['address'][0]}:{info['address'][1]}",
                'connected': info['connected']
            }
            for client_id, info in self.clients.items()
        ]
        self.mutex.unlock()
        return clients_list

    def subscribe_topic(self, client_id, topic):
        self.mutex.lock()
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = []
        
        if topic not in self.subscriptions[client_id]:
            self.subscriptions[client_id].append(topic)
        self.mutex.unlock()

    def unsubscribe_topic(self, client_id, topic):
        self.mutex.lock()
        if client_id in self.subscriptions and topic in self.subscriptions[client_id]:
            self.subscriptions[client_id].remove(topic)
        self.mutex.unlock()
        self.safe_log(f"客户端 {client_id} 取消订阅主题: {topic}")

    def publish_message(self, topic, message):
        """服务端主动发布消息"""
        self.mutex.lock()
        self.topics[topic] = message
        self.mutex.unlock()
        
        if topic != "siot/摄像头":
            self.safe_log(f"发布消息到主题 {topic}: {message[:50] if len(message) > 50 else message}")
        
        # 转发给订阅者
        if isinstance(message, str):
            payload = message.encode('utf-8')
        else:
            payload = message
        
        self.forward_to_subscribers(topic, payload)

    def stop(self):
        self.running = False
        
        self.mutex.lock()
        clients_to_disconnect = list(self.clients.keys())
        self.mutex.unlock()
        
        for client_id in clients_to_disconnect:
            self.disconnect_client(client_id)
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        self.process_queues()
        self.server_stopped.emit()
        self.safe_log("MQTT服务端已停止")

    def is_running(self):
        return self.running
