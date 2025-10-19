#!/usr/bin/env python3
"""
CarDD Dataset Integration Script
Downloads, preprocesses, and sets up CarDD dataset for training
"""

import os
import sys
import requests
import zipfile
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple
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
    
    def create_data_augmentation_script(self):
        """Create data augmentation script for better training"""
        augmentation_script = '''#!/usr/bin/env python3
"""
Data Augmentation Script for CarDD Dataset
Enhances dataset with various transformations
"""

import cv2
import numpy as np
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CarDDAugmentation:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.RandomShadow(p=0.3),
            A.RandomRain(p=0.2),
            A.RandomSunFlare(p=0.2),
        ])
    
    def augment_dataset(self, num_augmentations=3):
        """Apply augmentations to the dataset"""
        print(f"Applying {num_augmentations} augmentations per image...")
        
        for img_path in self.input_dir.glob("*.jpg"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            # Original image
            cv2.imwrite(str(self.output_dir / img_path.name), image)
            
            # Augmented images
            for i in range(num_augmentations):
                augmented = self.transform(image=image)['image']
                aug_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                cv2.imwrite(str(self.output_dir / aug_name), augmented)
        
        print(f"✅ Augmentation complete. Images saved to: {self.output_dir}")

if __name__ == "__main__":
    augmenter = CarDDAugmentation("data/cardd/images", "data/cardd/augmented")
    augmenter.augment_dataset()
'''
        
        script_path = self.data_dir / "augment_data.py"
        with open(script_path, 'w') as f:
            f.write(augmentation_script)
        
        print(f"✅ Data augmentation script created: {script_path}")
    
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

def export_model():
    """Export trained model to different formats"""
    model_path = "runs/train/cardd_damage_detection/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Export to different formats
    formats = ['onnx', 'torchscript', 'tflite']
    for fmt in formats:
        try:
            model.export(format=fmt)
            print(f"✅ Exported to {fmt} format")
        except Exception as e:
            print(f"❌ Failed to export to {fmt}: {e}")

if __name__ == "__main__":
    # Train the model
    results = train_cardd_model()
    
    # Export the model
    export_model()
    
    print("\\n🎉 CarDD model training complete!")
    print("\\n📁 Files created:")
    print("   • Best model: runs/train/cardd_damage_detection/weights/best.pt")
    print("   • Last model: runs/train/cardd_damage_detection/weights/last.pt")
    print("   • Training plots: runs/train/cardd_damage_detection/")
'''
        
        script_path = self.data_dir / "train_cardd.py"
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        print(f"✅ Training script created: {script_path}")
    
    def create_integration_script(self):
        """Create script to integrate trained model into the main app"""
        integration_script = '''#!/usr/bin/env python3
"""
Integration script to use trained CarDD model in the main application
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

class CarDDModel:
    def __init__(self, model_path: str = "models/cardd_best.pt"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained CarDD model"""
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"✅ CarDD model loaded from {self.model_path}")
            except Exception as e:
                print(f"❌ Failed to load CarDD model: {e}")
                self.model = None
        else:
            print(f"❌ CarDD model not found at {self.model_path}")
            print("Please train the model first using train_cardd.py")
    
    def detect_damage(self, image_bytes: bytes):
        """Detect damage using trained CarDD model"""
        if self.model is None:
            return None
        
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Run inference
            results = self.model(image)
            
            # Process results
            damage_regions = []
            damage_types = {
                'dent': 0,
                'scratch': 0,
                'crack': 0,
                'paint_damage': 0,
                'bumper_damage': 0,
                'glass_damage': 0
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        # Only include high-confidence detections
                        if confidence > 0.5:
                            damage_types[class_name] += 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            damage_regions.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
            
            # Calculate severity based on damage types and regions
            total_damage = sum(damage_types.values())
            if total_damage == 0:
                severity = "minor"
            elif total_damage <= 2:
                severity = "moderate"
            else:
                severity = "severe"
            
            return {
                'severity': severity,
                'damage_types': damage_types,
                'regions': damage_regions,
                'total_damage': total_damage
            }
            
        except Exception as e:
            print(f"❌ Error in CarDD detection: {e}")
            return None

# Integration with existing damage detection
def integrate_cardd_model():
    """Integrate CarDD model into the existing system"""
    
    # Update the damage detection module
    integration_code = '''
# Add this to app/cv/damage.py

from .cardd_model import CarDDModel

# Initialize CarDD model
cardd_model = CarDDModel()

def detect_damage_with_cardd(image_bytes: bytes) -> DamageResult:
    """Enhanced damage detection using CarDD model"""
    
    # Try CarDD model first
    cardd_result = cardd_model.detect_damage(image_bytes)
    
    if cardd_result:
        # Convert CarDD result to DamageResult format
        return DamageResult(
            annotated_bgr=annotate_image_with_cardd(image_bytes, cardd_result),
            severity=cardd_result['severity'],
            num_regions=len(cardd_result['regions']),
            boxes=[region['bbox'] for region in cardd_result['regions']],
            damage_types=cardd_result['damage_types'],
            total_damage_area=calculate_damage_area(cardd_result['regions'])
        )
    
    # Fallback to original method if CarDD fails
    return detect_damage(image_bytes)
'''
    
    print("✅ Integration code ready")
    print("Add the above code to your damage detection module")

if __name__ == "__main__":
    # Test the integration
    model = CarDDModel()
    integrate_cardd_model()
'''
        
        script_path = self.data_dir / "integrate_cardd.py"
        with open(script_path, 'w') as f:
            f.write(integration_script)
        
        print(f"✅ Integration script created: {script_path}")
    
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
        self.create_data_augmentation_script()
        self.create_training_script()
        self.create_integration_script()
        self.create_requirements_file()
        
        print("\n🎉 CarDD setup complete!")
        print("\n📋 Next steps:")
        print("1. Download CarDD dataset from: https://cardd-ustc.github.io/")
        print("2. Extract to: data/cardd/")
        print("3. Install requirements: pip install -r data/cardd/requirements_cardd.txt")
        print("4. Run training: python data/cardd/train_cardd.py")
        print("5. Integrate model: python data/cardd/integrate_cardd.py")

if __name__ == "__main__":
    setup = CarDDSetup()
    setup.run_setup()
