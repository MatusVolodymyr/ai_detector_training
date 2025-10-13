#!/usr/bin/env python3
"""
Validation script to check if the pipeline is properly set up
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Warning: Python 3.8+ recommended")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'torch', 'transformers', 'datasets', 'xgboost',
        'sklearn', 'spacy', 'optuna', 'yaml', 'numpy'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ No GPU detected - training will be slow!")
            return False
    except:
        return False

def check_spacy_models():
    """Check if spaCy models are installed"""
    try:
        import spacy
        models = ['en_core_web_sm', 'xx_sent_ud_sm']
        all_installed = True
        
        for model in models:
            try:
                spacy.load(model)
                print(f"✓ spaCy model: {model}")
            except:
                print(f"✗ spaCy model: {model} - MISSING")
                all_installed = False
        
        if not all_installed:
            print("\nRun: python -m spacy download en_core_web_sm xx_sent_ud_sm")
            return False
        return True
    except:
        return False

def check_directories():
    """Check if necessary directories exist"""
    dirs = ['modules', 'utils']
    all_exist = True
    
    for directory in dirs:
        if os.path.exists(directory):
            print(f"✓ Directory: {directory}")
        else:
            print(f"✗ Directory: {directory} - MISSING")
            all_exist = False
    
    return all_exist

def check_config():
    """Check if config file exists and is valid"""
    if not os.path.exists('config.yaml'):
        print("✗ config.yaml not found")
        return False
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['dataset', 'data_split', 'deberta', 'xgboost']
        for key in required_keys:
            if key not in config:
                print(f"✗ config.yaml missing key: {key}")
                return False
        
        print("✓ config.yaml is valid")
        return True
    except Exception as e:
        print(f"✗ Error reading config.yaml: {e}")
        return False

def check_parent_modules():
    """Check if parent modules are accessible"""
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, parent_dir)
    
    modules = [
        'modules.stylometric_extraction',
        'modules.incremental_processor',
        'modules.prediction_service_dataset'
    ]
    
    all_found = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ Parent module: {module}")
        except Exception as e:
            print(f"✗ Parent module: {module} - ERROR: {str(e)[:50]}")
            all_found = False
    
    return all_found

def main():
    """Run all validation checks"""
    print("="*60)
    print("AI Detector Training Pipeline - Validation")
    print("="*60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("spaCy Models", check_spacy_models),
        ("Directories", check_directories),
        ("Config File", check_config),
        ("Parent Modules", check_parent_modules),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} - {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Update config.yaml with your dataset path")
        print("3. Run: python main.py --config config.yaml")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor setup help, see README.md or run: ./setup.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
