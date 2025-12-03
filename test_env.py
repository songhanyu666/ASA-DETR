"""
环境测试脚本 - Environment Test Script
用于验证ASA-DETR运行环境是否正确配置
"""

import sys
import platform

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本 / Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 10:
        print("✓ Python版本符合要求 (3.8-3.10) / Python version is compatible")
        return True
    else:
        print("✗ Python版本不符合要求，需要3.8-3.10 / Python version incompatible, requires 3.8-3.10")
        return False

def check_pytorch():
    """检查PyTorch安装"""
    try:
        import torch
        print(f"✓ PyTorch版本 / PyTorch version: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA可用 / CUDA available: True")
            print(f"  CUDA版本 / CUDA version: {torch.version.cuda}")
            print(f"  GPU设备数量 / Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    显存 / Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠ CUDA不可用，将使用CPU训练（速度较慢）")
            print("  CUDA not available, will use CPU for training (slower)")
        
        return True
    except ImportError:
        print("✗ PyTorch未安装 / PyTorch not installed")
        print("  请运行: pip install torch torchvision torchaudio")
        print("  Please run: pip install torch torchvision torchaudio")
        return False

def check_dependencies():
    """检查其他依赖包"""
    dependencies = {
        'numpy': 'NumPy',
        'opencv-cv2': 'OpenCV',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'tensorboard': 'TensorBoard'
    }
    
    all_installed = True
    
    for module, name in dependencies.items():
        try:
            if module == 'opencv-cv2':
                import cv2
                print(f"✓ {name}: {cv2.__version__}")
            elif module == 'PIL':
                from PIL import Image
                print(f"✓ {name}: {Image.__version__}")
            elif module == 'yaml':
                import yaml
                print(f"✓ {name}: {yaml.__version__}")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}未安装 / {name} not installed")
            all_installed = False
    
    return all_installed

def check_system_info():
    """显示系统信息"""
    print("\n" + "="*60)
    print("系统信息 / System Information")
    print("="*60)
    print(f"操作系统 / OS: {platform.system()} {platform.release()}")
    print(f"处理器 / Processor: {platform.processor()}")
    print(f"架构 / Architecture: {platform.machine()}")
    print("="*60 + "\n")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("ASA-DETR 环境检查 / Environment Check")
    print("="*60 + "\n")
    
    # 检查系统信息
    check_system_info()
    
    # 检查Python版本
    print("1. 检查Python版本 / Checking Python version...")
    python_ok = check_python_version()
    print()
    
    # 检查PyTorch
    print("2. 检查PyTorch / Checking PyTorch...")
    pytorch_ok = check_pytorch()
    print()
    
    # 检查其他依赖
    print("3. 检查依赖包 / Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    # 总结
    print("="*60)
    print("检查结果总结 / Summary")
    print("="*60)
    
    if python_ok and pytorch_ok and deps_ok:
        print("✓ 所有检查通过！环境配置正确。")
        print("✓ All checks passed! Environment is properly configured.")
        print("\n您可以开始使用ASA-DETR了！")
        print("You can start using ASA-DETR now!")
        print("\n快速开始:")
        print("Quick start:")
        print("  python train.py --cfg configs/asa-detr.yaml --data datasets/RSLD-2K/data.yaml")
        return 0
    else:
        print("✗ 部分检查未通过，请安装缺失的依赖包。")
        print("✗ Some checks failed, please install missing dependencies.")
        print("\n安装所有依赖:")
        print("Install all dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)