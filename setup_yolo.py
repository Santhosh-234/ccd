#!/usr/bin/env python3
"""
YOLO Model Setup Script for Multimeta Car Damage Detection
Downloads and sets up YOLO model for better damage detection accuracy
"""

import os
import sys
import subprocess
from pathlib import Path

def install_ultralytics():
    """Install ultralytics package"""
    print("📦 Installing ultralytics...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
        print("✅ ultralytics installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install ultralytics: {e}")
        return False

def download_yolo_model():
    """Download YOLO model"""
    print("🤖 Downloading YOLO model...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "car_damage.pt"
    
    if model_path.exists():
        print(f"✅ YOLO model already exists at {model_path}")
        return str(model_path)
    
    try:
        # Import and download YOLO model
        from ultralytics import YOLO
        
        print("Downloading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        model.save(str(model_path))
        
        print(f"✅ YOLO model downloaded successfully to {model_path}")
        return str(model_path)
        
    except ImportError:
        print("❌ ultralytics not installed. Installing now...")
        if install_ultralytics():
            return download_yolo_model()
        else:
            return None
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")
        return None

def update_env_file(model_path):
    """Update .env file with YOLO model path"""
    print("🔧 Updating environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    # Create .env from .env.example if it doesn't exist
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from .env.example...")
        env_file.write_text(env_example.read_text())
    
    if not env_file.exists():
        print("❌ .env file not found. Please create it manually.")
        return False
    
    # Read current .env content
    content = env_file.read_text()
    
    # Update or add YOLO_MODEL_PATH
    lines = content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if line.startswith('YOLO_MODEL_PATH='):
            lines[i] = f'YOLO_MODEL_PATH={model_path}'
            updated = True
            break
    
    if not updated:
        lines.append(f'YOLO_MODEL_PATH={model_path}')
    
    # Write back to .env
    env_file.write_text('\n'.join(lines))
    print(f"✅ Updated .env file with YOLO_MODEL_PATH={model_path}")
    return True

def verify_setup():
    """Verify the setup is working"""
    print("🔍 Verifying setup...")
    
    try:
        # Test import
        from ultralytics import YOLO
        print("✅ ultralytics import successful")
        
        # Test model loading
        model_path = Path("models/car_damage.pt")
        if model_path.exists():
            model = YOLO(str(model_path))
            print("✅ YOLO model loads successfully")
            return True
        else:
            print("❌ YOLO model file not found")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("🚀 Setting up YOLO model for Multimeta Car Damage Detection")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("app").exists() or not Path("requirements.txt").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Download YOLO model
    model_path = download_yolo_model()
    if not model_path:
        print("❌ Failed to download YOLO model")
        print("💡 The system will still work with traditional CV fallback")
        sys.exit(1)
    
    # Update environment file
    if not update_env_file(model_path):
        print("⚠️  Failed to update .env file, but model is ready")
    
    # Verify setup
    if verify_setup():
        print("\n🎉 YOLO setup completed successfully!")
        print("You can now run the application with enhanced damage detection.")
        print("\nNext steps:")
        print("1. Run: python deploy.py local")
        print("2. Or: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("3. Open: http://localhost:8000")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("The system will use traditional CV methods as fallback")

if __name__ == "__main__":
    main()
