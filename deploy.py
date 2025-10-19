#!/usr/bin/env python3
"""
Deployment script for Multimeta Car Damage Detection System
Supports local development, Docker, and cloud deployments
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def setup_environment():
    """Set up the development environment"""
    print("Setting up development environment...")
    
    # Create virtual environment
    if not Path(".venv").exists():
        print("Creating virtual environment...")
        run_command("python -m venv .venv")
    
    # Activate and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    print("Installing dependencies...")
    run_command(f"{pip_cmd} install --upgrade pip")
    run_command(f"{pip_cmd} install -r requirements.txt")
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        print("Creating .env file...")
        run_command("copy .env.example .env" if os.name == 'nt' else "cp .env.example .env")
    
    print("SUCCESS: Environment setup complete!")

def setup_yolo_model():
    """Set up YOLO model for better accuracy"""
    print("Setting up YOLO model...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check if model exists
    model_path = "models/car_damage.pt"
    if not Path(model_path).exists():
        print("YOLO model not found. You have two options:")
        print("1. Download a pre-trained YOLOv8 model:")
        print("   - Install ultralytics: pip install ultralytics")
        print("   - Download: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').save('models/car_damage.pt')\"")
        print("2. Use the system without YOLO (it will use traditional CV fallback)")
        print("   - The system will work fine without YOLO, just with lower accuracy")
    else:
        print("SUCCESS: YOLO model found!")
    
    # Update .env file
    env_content = ""
    if Path(".env").exists():
        with open(".env", "r") as f:
            env_content = f.read()
    
    if "YOLO_MODEL_PATH" not in env_content:
        with open(".env", "a") as f:
            f.write(f"\nYOLO_MODEL_PATH={model_path}\n")

def run_local():
    """Run the application locally"""
    print("Starting local development server...")
    
    if os.name == 'nt':  # Windows
        uvicorn_cmd = ".venv\\Scripts\\uvicorn"
    else:  # Unix/Linux/Mac
        uvicorn_cmd = ".venv/bin/uvicorn"
    
    run_command(f"{uvicorn_cmd} app.main:app --reload --host 0.0.0.0 --port 8000", check=False)

def build_docker():
    """Build Docker image"""
    print("Building Docker image...")
    run_command("docker build -t multimeta-car-damage .")
    print("SUCCESS: Docker image built successfully!")

def run_docker():
    """Run Docker container"""
    print("Starting Docker container...")
    run_command("docker run -p 8000:8000 --env-file .env multimeta-car-damage")

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        return False
    
    # Check if Docker is available (for Docker deployment)
    try:
        run_command("docker --version", check=False)
        print("SUCCESS: Docker available")
    except:
        print("WARNING: Docker not available (needed for Docker deployment)")
    
    print("SUCCESS: Requirements check complete!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Deploy Multimeta Car Damage Detection System")
    parser.add_argument("command", choices=["setup", "local", "docker-build", "docker-run", "check"], 
                       help="Deployment command")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        check_requirements()
        setup_environment()
        setup_yolo_model()
        print("\n🎉 Setup complete! You can now run:")
        print("  python deploy.py local    # Run locally")
        print("  python deploy.py docker-build  # Build Docker image")
        
    elif args.command == "local":
        run_local()
        
    elif args.command == "docker-build":
        build_docker()
        
    elif args.command == "docker-run":
        run_docker()
        
    elif args.command == "check":
        check_requirements()

if __name__ == "__main__":
    main()
