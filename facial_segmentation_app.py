import os
import cv2
import sys
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import transforms
import threading
import webbrowser

# Import model architectures from segmentation scripts
from eye_segmentation import UNet as EyeUNet
from nose_segmentation import UNet as NoseUNet
from lips_segmentation import UNet as LipsUNet

class FacialSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Feature Segmentation")
        self.root.geometry("1280x800")
        self.root.minsize(1280, 800)
        
        # Set theme and styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 12), padding=5)
        self.style.configure('TLabel', font=('Arial', 12))
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Feature.TCheckbutton', font=('Arial', 12))
        
        # Variables
        self.image_path = ""
        self.original_image = None
        self.tk_image = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature selection variables
        self.show_eyes = tk.BooleanVar(value=True)
        self.show_nose = tk.BooleanVar(value=True)
        self.show_lips = tk.BooleanVar(value=True)
        
        # Output type selection
        self.output_type = tk.StringVar(value="overlay")
        
        # Models and masks
        self.eye_model = None
        self.nose_model = None
        self.lips_model = None
        self.eye_mask = None
        self.nose_mask = None
        self.lips_mask = None
        self.combined_mask = None
        self.processed_image = None
        self.segmentation_complete = False
        
        # Create GUI elements
        self.create_menu()
        self.create_main_frame()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load models in background
        self.load_button.config(state=tk.DISABLED)
        self.segment_button.config(state=tk.DISABLED)
        self.status_var.set("Loading models...")
        
        # Start a thread to load models
        threading.Thread(target=self.load_models, daemon=True).start()
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save Result", command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
    
    def create_main_frame(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Header
        header_label = ttk.Label(controls_frame, text="Facial Feature Segmentation", style='Header.TLabel')
        header_label.pack(side=tk.TOP, pady=5)
        
        # Load image button
        self.load_button = ttk.Button(controls_frame, text="Load Image", command=self.open_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Segment button
        self.segment_button = ttk.Button(controls_frame, text="Run Segmentation", command=self.run_segmentation)
        self.segment_button.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_button = ttk.Button(controls_frame, text="Save Result", command=self.save_result)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_button.config(state=tk.DISABLED)
        
        # Feature selection
        feature_frame = ttk.LabelFrame(controls_frame, text="Features to Segment")
        feature_frame.pack(side=tk.LEFT, padx=20)
        
        eye_check = ttk.Checkbutton(feature_frame, text="Eyes", variable=self.show_eyes, 
                                    style='Feature.TCheckbutton', command=self.update_display)
        eye_check.pack(side=tk.LEFT, padx=10)
        
        nose_check = ttk.Checkbutton(feature_frame, text="Nose", variable=self.show_nose, 
                                     style='Feature.TCheckbutton', command=self.update_display)
        nose_check.pack(side=tk.LEFT, padx=10)
        
        lips_check = ttk.Checkbutton(feature_frame, text="Lips", variable=self.show_lips, 
                                     style='Feature.TCheckbutton', command=self.update_display)
        lips_check.pack(side=tk.LEFT, padx=10)
        
        # Output type selection
        output_frame = ttk.LabelFrame(controls_frame, text="Output Type")
        output_frame.pack(side=tk.LEFT, padx=20)
        
        overlay_radio = ttk.Radiobutton(output_frame, text="Color Overlay", 
                                       variable=self.output_type, value="overlay", 
                                       command=self.update_display)
        overlay_radio.pack(side=tk.LEFT, padx=10)
        
        mask_radio = ttk.Radiobutton(output_frame, text="Binary Mask", 
                                     variable=self.output_type, value="mask", 
                                     command=self.update_display)
        mask_radio.pack(side=tk.LEFT, padx=10)
        
        # Image display area
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Original image
        original_frame = ttk.LabelFrame(display_frame, text="Original Image")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg="lightgray")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Result image
        result_frame = ttk.LabelFrame(display_frame, text="Segmentation Result")
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.result_canvas = tk.Canvas(result_frame, bg="lightgray")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
    
    def load_models(self):
        try:
            # Eye model
            self.eye_model = EyeUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(self.device)
            self.eye_model.load_state_dict(torch.load('output/eye_segmentation_model.pth', map_location=self.device))
            self.eye_model.eval()
            
            # Nose model
            self.nose_model = NoseUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(self.device)
            self.nose_model.load_state_dict(torch.load('output_nose/nose_segmentation_model.pth', map_location=self.device))
            self.nose_model.eval()
            
            # Lips model
            self.lips_model = LipsUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(self.device)
            self.lips_model.load_state_dict(torch.load('output_lips/lips_segmentation_model.pth', map_location=self.device))
            self.lips_model.eval()
            
            # Update UI from the main thread
            self.root.after(0, self.models_loaded)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
    
    def models_loaded(self):
        self.load_button.config(state=tk.NORMAL)
        self.status_var.set("Models loaded. Ready to process images.")
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image", 
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert('RGB')
            
            # Reset segmentation results
            self.eye_mask = None
            self.nose_mask = None
            self.lips_mask = None
            self.combined_mask = None
            self.processed_image = None
            self.segmentation_complete = False
            
            # Display the original image
            self.display_original_image()
            
            # Enable segmentation button
            self.segment_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
            
            # Clear result canvas
            self.result_canvas.delete("all")
            
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def display_original_image(self):
        if self.original_image:
            # Get canvas dimensions
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not fully initialized, try again later
                self.root.after(100, self.display_original_image)
                return
            
            # Resize image to fit canvas while maintaining aspect ratio
            img_width, img_height = self.original_image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_img)
            
            # Center image in canvas
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2
            
            # Clear previous image and display new one
            self.original_canvas.delete("all")
            self.original_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_image)
    
    def run_segmentation(self):
        if not self.original_image:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        self.status_var.set("Running segmentation...")
        self.segment_button.config(state=tk.DISABLED)
        
        # Start a thread for segmentation
        threading.Thread(target=self.segment_image, daemon=True).start()
    
    def segment_image(self):
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(self.original_image)
            
            # Run segmentation models
            if self.show_eyes.get():
                self.eye_mask = self.predict_mask(self.eye_model, img_tensor)
            else:
                self.eye_mask = np.zeros((128, 128), dtype=np.uint8)
                
            if self.show_nose.get():
                self.nose_mask = self.predict_mask(self.nose_model, img_tensor)
            else:
                self.nose_mask = np.zeros((128, 128), dtype=np.uint8)
                
            if self.show_lips.get():
                self.lips_mask = self.predict_mask(self.lips_model, img_tensor)
            else:
                self.lips_mask = np.zeros((128, 128), dtype=np.uint8)
            
            # Create combined mask
            self.combined_mask = np.maximum(np.maximum(self.eye_mask, self.nose_mask), self.lips_mask)
            
            # Set flag to indicate segmentation is complete
            self.segmentation_complete = True
            
            # Update UI from the main thread
            self.root.after(0, self.segmentation_complete_callback)
            
        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed: {str(e)}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.segment_button.config(state=tk.NORMAL))
    
    def segmentation_complete_callback(self):
        self.update_display()
        self.segment_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.status_var.set("Segmentation complete!")
    
    def update_display(self):
        if not self.segmentation_complete:
            return
        
        # Get canvas dimensions
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not fully initialized, try again later
            self.root.after(100, self.update_display)
            return
        
        # Create the appropriate output based on selection
        if self.output_type.get() == "overlay":
            result_img = self.create_overlay_image()
        else:  # mask
            result_img = self.create_mask_image()
        
        # Resize to fit canvas
        img_width, img_height = result_img.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        resized_img = result_img.resize((new_width, new_height), Image.LANCZOS)
        self.processed_image = ImageTk.PhotoImage(resized_img)
        
        # Center image in canvas
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        
        # Clear previous image and display new one
        self.result_canvas.delete("all")
        self.result_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.processed_image)
    
    def create_overlay_image(self):
        # Convert PIL image to numpy array
        img_np = np.array(self.original_image)
        
        # Resize masks to match original image dimensions
        h, w = img_np.shape[:2]
        
        # Create colored masks with their respective colors
        eye_color = (0, 0, 255)     # Blue for eyes
        nose_color = (255, 165, 0)  # Orange for nose
        lips_color = (128, 0, 128)  # Purple for lips
        
        # Create colored overlay
        colored_mask = np.zeros_like(img_np)
        
        if self.show_eyes.get() and self.eye_mask is not None:
            eye_mask_resized = cv2.resize(self.eye_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            for c in range(3):
                colored_mask[:, :, c] += eye_mask_resized * eye_color[c]
        
        if self.show_nose.get() and self.nose_mask is not None:
            nose_mask_resized = cv2.resize(self.nose_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            for c in range(3):
                colored_mask[:, :, c] += nose_mask_resized * nose_color[c]
        
        if self.show_lips.get() and self.lips_mask is not None:
            lips_mask_resized = cv2.resize(self.lips_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            for c in range(3):
                colored_mask[:, :, c] += lips_mask_resized * lips_color[c]
        
        # Blend the original image with the colored mask
        alpha = 0.4
        blended = cv2.addWeighted(img_np, 1, colored_mask, alpha, 0)
        
        # Convert back to PIL image
        return Image.fromarray(blended)
    
    def create_mask_image(self):
        # Get original image dimensions
        w, h = self.original_image.size
        
        # Create RGB mask
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Resize masks to match original image dimensions
        if self.show_eyes.get() and self.eye_mask is not None:
            eye_mask_resized = cv2.resize(self.eye_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            rgb_mask[eye_mask_resized == 1] = [0, 0, 255]  # Blue for eyes
        
        if self.show_nose.get() and self.nose_mask is not None:
            nose_mask_resized = cv2.resize(self.nose_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            rgb_mask[nose_mask_resized == 1] = [255, 165, 0]  # Orange for nose
        
        if self.show_lips.get() and self.lips_mask is not None:
            lips_mask_resized = cv2.resize(self.lips_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            rgb_mask[lips_mask_resized == 1] = [128, 0, 128]  # Purple for lips
        
        # Convert to PIL image
        return Image.fromarray(rgb_mask)
    
    def preprocess_image(self, image, size=128):
        # Resize image
        img_resized = image.resize((size, size))
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        return img_tensor
    
    def predict_mask(self, model, img_tensor):
        # Set model to evaluation mode
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            predicted_mask = torch.sigmoid(output) > 0.5
        
        # Convert to numpy array
        predicted_mask = predicted_mask.cpu().squeeze().numpy().astype(np.uint8)
        
        return predicted_mask
    
    def save_result(self):
        if not self.segmentation_complete or not self.processed_image:
            messagebox.showwarning("Nothing to Save", "Please run segmentation first.")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Result", 
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Save the current result
            if self.output_type.get() == "overlay":
                result_img = self.create_overlay_image()
            else:
                result_img = self.create_mask_image()
            
            result_img.save(file_path)
            self.status_var.set(f"Result saved to: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def show_about(self):
        about_text = """
        Facial Feature Segmentation App
        
        This application uses deep learning models to segment facial features
        such as eyes, nose, and lips from images.
        
        Models trained on the CelebAMask-HQ dataset.
        
        Created with PyTorch and Tkinter.
        """
        
        messagebox.showinfo("About", about_text)
    
    def show_user_guide(self):
        guide_text = """
        User Guide:
        
        1. Click 'Load Image' to open a face image
        2. Click 'Run Segmentation' to process the image
        3. Use the checkboxes to show/hide specific features
        4. Choose between 'Color Overlay' or 'Binary Mask' output
        5. Click 'Save Result' to save the current view
        
        Models:
        - Eyes segmentation (blue)
        - Nose segmentation (orange)
        - Lips segmentation (purple)
        """
        
        messagebox.showinfo("User Guide", guide_text)


# Function to launch the application
def main():
    root = tk.Tk()
    app = FacialSegmentationApp(root)
    
    # Add app icon if available
    try:
        if getattr(sys, 'frozen', False):
            # If running as packaged executable
            application_path = sys._MEIPASS
        else:
            # If running as script
            application_path = os.path.dirname(os.path.abspath(__file__))
            
        icon_path = os.path.join(application_path, 'app_icon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass
    
    # Start event loop
    root.mainloop()


if __name__ == "__main__":
    main() 