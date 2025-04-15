import os
import sys
import shutil
import subprocess
import platform
import argparse

def create_icon():
    """Create a simple icon if one doesn't exist"""
    try:
        from PIL import Image, ImageDraw
        
        if os.path.exists('app_icon.ico'):
            print("Icon already exists")
            return
            
        # Create a simple colored icon
        icon_size = 256
        img = Image.new('RGB', (icon_size, icon_size), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Draw a face outline
        draw.ellipse((40, 40, icon_size-40, icon_size-40), outline=(60, 60, 60), width=8)
        
        # Draw eyes (blue)
        eye_color = (0, 0, 255)
        draw.ellipse((80, 90, 110, 120), fill=eye_color)
        draw.ellipse((icon_size-110, 90, icon_size-80, 120), fill=eye_color)
        
        # Draw nose (orange)
        nose_color = (255, 165, 0)
        draw.polygon([(icon_size//2, 130), (icon_size//2-20, 170), (icon_size//2+20, 170)], fill=nose_color)
        
        # Draw mouth (purple)
        lips_color = (128, 0, 128)
        draw.arc((80, 120, icon_size-80, 200), 0, 180, fill=lips_color, width=8)
        
        # Save as ICO for Windows
        img.save('app_icon.ico', format='ICO')
        
        # If on macOS, also save as PNG for the app bundle
        if platform.system() == 'Darwin':
            img.save('app_icon.png', format='PNG')
            
        print("Created app icon")
    except Exception as e:
        print(f"Error creating icon: {e}")
        print("Continuing without icon...")

def package_application(onefile=True, debug=False):
    """Package the application using PyInstaller"""
    print("Preparing to package application...")
    
    # Create the icon
    create_icon()
    
    # Define PyInstaller command
    pyinstaller_cmd = [
        'pyinstaller',
        '--name=Facial_Segmentation_App',
        '--noconfirm',
        '--clean',
    ]
    
    # Add one-file or one-dir option
    if onefile:
        pyinstaller_cmd.append('--onefile')
    else:
        pyinstaller_cmd.append('--onedir')
    
    # Add windowed mode (no console) if not debugging
    if not debug:
        pyinstaller_cmd.append('--windowed')
    
    # Add icon if available
    if os.path.exists('app_icon.ico'):
        pyinstaller_cmd.append(f'--icon=app_icon.ico')
    
    # Add model files as data
    model_paths = [
        'output/eye_segmentation_model.pth',
        'output_nose/nose_segmentation_model.pth',
        'output_lips/lips_segmentation_model.pth'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            pyinstaller_cmd.append(f'--add-data={model_path}{os.pathsep}{os.path.dirname(model_path)}')
        else:
            print(f"Warning: Model file {model_path} not found. Make sure to train models before packaging.")
    
    # Add the main script
    pyinstaller_cmd.append('facial_segmentation_app.py')
    
    print(f"Running PyInstaller with command: {' '.join(pyinstaller_cmd)}")
    
    # Run PyInstaller
    try:
        subprocess.run(pyinstaller_cmd, check=True)
        print("\nPackaging complete!")
        
        # Output location info
        if onefile:
            exe_path = os.path.join('dist', 'Facial_Segmentation_App')
            if platform.system() == 'Windows':
                exe_path += '.exe'
            print(f"\nExecutable created at: {exe_path}")
        else:
            dist_path = os.path.join('dist', 'Facial_Segmentation_App')
            print(f"\nApplication folder created at: {dist_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during packaging: {e}")
        print("Make sure PyInstaller is installed (`pip install pyinstaller`)")
        return False
    
    return True

def verify_requirements():
    """Verify that all required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pillow', 'opencv-python', 
        'matplotlib', 'pyinstaller'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.split('-')[0])  # Handle packages like opencv-python
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        else:
            print("Please install the missing packages before continuing.")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Package the Facial Segmentation App')
    parser.add_argument('--onedir', action='store_true', help='Create a directory instead of a single file')
    parser.add_argument('--debug', action='store_true', help='Enable console for debugging')
    args = parser.parse_args()
    
    print("======= Facial Segmentation App Packager =======")
    
    # Verify requirements
    if not verify_requirements():
        return
    
    # Package the application
    package_application(onefile=not args.onedir, debug=args.debug)

if __name__ == '__main__':
    main() 