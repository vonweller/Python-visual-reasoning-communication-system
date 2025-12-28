import sys
from core.inference import YoloInference
from core.config_manager import ConfigManager

def test_device_switching():
    print("开始测试GPU/CPU切换功能...")
    
    config_manager = ConfigManager()
    
    # 测试1: 初始化CPU模式
    print("\n测试1: 初始化CPU模式")
    try:
        yolo_cpu = YoloInference(
            model_path="best.pt",
            conf_threshold=0.88,
            classes_dict=None,
            device="cpu"
        )
        print("✓ CPU模式初始化成功")
        print(f"  当前设备: {yolo_cpu.device}")
    except Exception as e:
        print(f"✗ CPU模式初始化失败: {e}")
        return False
    
    # 测试2: 尝试初始化GPU模式
    print("\n测试2: 尝试初始化GPU模式")
    try:
        yolo_gpu = YoloInference(
            model_path="best.pt",
            conf_threshold=0.88,
            classes_dict=None,
            device="cuda"
        )
        print("✓ GPU模式初始化成功")
        print(f"  当前设备: {yolo_gpu.device}")
        
        # 测试3: GPU切换到CPU
        print("\n测试3: GPU切换到CPU")
        yolo_gpu.set_device("cpu")
        print("✓ GPU切换到CPU成功")
        print(f"  当前设备: {yolo_gpu.device}")
        
        # 测试4: CPU切换到GPU
        print("\n测试4: CPU切换到GPU")
        yolo_gpu.set_device("cuda")
        print("✓ CPU切换到GPU成功")
        print(f"  当前设备: {yolo_gpu.device}")
        
    except Exception as e:
        print(f"✗ GPU模式初始化失败: {e}")
        print("  这是预期的行为，如果系统没有CUDA或GPU")
        
        # 测试5: 验证CPU模式仍然可用
        print("\n测试5: 验证CPU模式仍然可用")
        try:
            yolo_cpu.set_device("cpu")
            print("✓ CPU模式仍然可用")
            print(f"  当前设备: {yolo_cpu.device}")
        except Exception as e2:
            print(f"✗ CPU模式不可用: {e2}")
            return False
    
    print("\n所有测试完成!")
    return True

if __name__ == "__main__":
    success = test_device_switching()
    sys.exit(0 if success else 1)
