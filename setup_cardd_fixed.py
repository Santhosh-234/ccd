#!/usr/bin/env python3
"""
CarDD Dataset Integration Script - Fixed Version
Downloads, preprocesses, and sets up CarDD dataset for training
"""

import os
import sys
from pathlib import Path
import yaml

class CarDDSetup:
    def __init__(self, data_dir: str = "data/cardd"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.yolo_dir = self.data_dir / "yolo_format"
        
    def setup_directories(self):
        """Create necessary directories"""
        print("Setting up CarDD directories...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        self.yolo_dir.mkdir(exist_ok=True)
        print("✅ Directories created successfully")
    
    def download_cardd_info(self):
        """Display information about CarDD dataset"""
        print("\n" + "="*60)
        print("🚗 CarDD Dataset Information")
        print("="*60)
        print("📊 Dataset Details:")
        print("   • 4,000 high-resolution images")
        print("   • 9,000+ annotated damage instances")
        print("   • 6 damage categories: dents, scratches, cracks, etc.")
        print("   • Resolution: Various (typically 1920x1080)")
        print("\n🔗 Access Information:")
        print("   • Website: https://cardd-ustc.github.io/")
        print("   • License: Academic/Research use")
        print("   • Download: Requires registration form")
        print("\n📋 Steps to get the dataset:")
        print("   1. Visit: https://cardd-ustc.github.io/")
        print("   2. Fill out the licensing form")
        print("   3. Download the dataset")
        print("   4. Extract to: data/cardd/")
        print("="*60)
    
    def create_yolo_structure(self):
        """Create YOLO format directory structure"""
        print("\nCreating YOLO format structure...")
        
        # Create YOLO directories
        (self.yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        print("✅ YOLO structure created")
    
    def create_yolo_config(self):
        """Create YOLO configuration file"""
        config = {
            'path': str(self.yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 6,  # Number of classes
            'names': [
                'dent',
                'scratch', 
                'crack',
                'paint_damage',
                'bumper_damage',
                'glass_damage'
            ]
        }
        
        config_path = self.yolo_dir / "cardd.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ YOLO config created: {config_path}")
        return config_path
    
    def create_training_script(self):
        """Create YOLO training script"""
        training_script = '''#!/usr/bin/env python3
"""
YOLO Training Script for CarDD Dataset
Trains YOLOv8 model on CarDD dataset
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def train_cardd_model():
    """Train YOLO model on CarDD dataset"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Start with nano model for faster training
    
    # Training parameters
    training_args = {
        'data': 'data/cardd/yolo_format/cardd.yaml',
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'device': device,
        'project': 'runs/train',
        'name': 'cardd_damage_detection',
        'save_period': 10,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'augment': True,
        'mixup': 0.15,
        'copy_paste': 0.3,
    }
    
    print("🚀 Starting YOLO training...")
    print(f"Training parameters: {training_args}")
    
    # Start training
    results = model.train(**training_args)
    
    print("✅ Training completed!")
    print(f"Best model saved to: {results.save_dir}")
    
    # Validate the model
    print("\\n🔍 Validating model...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.3f}")
    print(f"Validation mAP50-95: {metrics.box.map:.3f}")
    
    return results

if __name__ == "__main__":
    # Train the model
    results = train_cardd_model()
    print("\\n🎉 CarDD model training complete!")
'''
        
        script_path = self.data_dir / "train_cardd.py"
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        print(f"✅ Training script created: {script_path}")
    
    def create_requirements_file(self):
        """Create requirements file for CarDD training"""
        requirements = '''# CarDD Dataset Training Requirements
ultralytics>=8.0.0
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
albumentations>=1.3.0
numpy>=1.21.0
Pillow>=8.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
pyyaml>=6.0
'''
        
        req_path = self.data_dir / "requirements_cardd.txt"
        with open(req_path, 'w') as f:
            f.write(requirements)
        
        print(f"✅ Requirements file created: {req_path}")
    
    def run_setup(self):
        """Run the complete CarDD setup"""
        print("🚗 Setting up CarDD Dataset Integration")
        print("="*50)
        
        # Setup directories
        self.setup_directories()
        
        # Show dataset info
        self.download_cardd_info()
        
        # Create YOLO structure
        self.create_yolo_structure()
        
        # Create configuration
        config_path = self.create_yolo_config()
        
        # Create scripts
        self.create_training_script()
        self.create_requirements_file()
        
        print("\n🎉 CarDD setup complete!")
        print("\n📋 Next steps:")
        print("1. Download CarDD dataset from: https://cardd-ustc.github.io/")
        print("2. Extract to: data/cardd/")
        print("3. Install requirements: pip install -r data/cardd/requirements_cardd.txt")
        print("4. Run training: python data/cardd/train_cardd.py")
        print("5. Copy trained model to: models/cardd_best.pt")

if __name__ == "__main__":
    setup = CarDDSetup()
    setup.run_setup()
