import json
import os
import sys

class ConfigManager:
    def __init__(self, config_path="config/config.json", classes_path="config/classes.json"):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.config_path = os.path.join(base_path, config_path)
        self.classes_path = os.path.join(base_path, classes_path)
        self.config = self.load_json(self.config_path)
        self.classes = self.load_json(self.classes_path)

    def load_json(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return {}

    def save_config(self):
        self.save_json(self.config_path, self.config)

    def save_classes(self):
        self.save_json(self.classes_path, self.classes)

    def save_json(self, path, data):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving {path}: {e}")

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def set(self, key, value):
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
        self.save_config()
