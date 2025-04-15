import os
import cv2
import uuid
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torchvision.transforms as transforms
import sys
import json

# Add parent directory to path for importing models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import model architectures - wrapped in try/except to handle missing models
try:
    from eye_segmentation import UNet as EyeUNet
    from nose_segmentation import UNet as NoseUNet
    from lips_segmentation import UNet as LipsUNet
    models_importable = True
except ImportError as e:
    print(f"Error importing model definitions: {str(e)}")
    models_importable = False

app = Flask(__name__)
# Configure CORS for production
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://facial-segmentation-frontend.onrender.com",
            "http://localhost:5173",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Model paths - update these to match your actual file locations
MODEL_PATHS = {
    'eye': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output', 'eye_segmentation_model.pth'),
    'nose': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output_nose', 'nose_segmentation_model.pth'),
    'lips': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output_lips', 'lips_segmentation_model.pth')
}

# Initialize models
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature colors (RGB)
FEATURE_COLORS = {
    'eye': (0, 0, 255),     # Blue for eyes
    'nose': (255, 165, 0),  # Orange for nose
    'lips': (128, 0, 128)   # Purple for lips
}

def load_models():
    """Load all segmentation models"""
    if not models_importable:
        print("Cannot load models because model definitions couldn't be imported")
        return False
        
    try:
        models_loaded = 0
        # Try to load each model individually
        try:
            # Eye model
            print(f"Loading eye model from: {MODEL_PATHS['eye']}")
            if os.path.exists(MODEL_PATHS['eye']):
                eye_model = EyeUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(device)
                eye_model.load_state_dict(torch.load(MODEL_PATHS['eye'], map_location=device))
                eye_model.eval()
                models['eye'] = eye_model
                models_loaded += 1
            else:
                print(f"Eye model file not found at: {MODEL_PATHS['eye']}")
        except Exception as e:
            print(f"Error loading eye model: {str(e)}")
        
        try:
            # Nose model
            print(f"Loading nose model from: {MODEL_PATHS['nose']}")
            if os.path.exists(MODEL_PATHS['nose']):
                nose_model = NoseUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(device)
                nose_model.load_state_dict(torch.load(MODEL_PATHS['nose'], map_location=device))
                nose_model.eval()
                models['nose'] = nose_model
                models_loaded += 1
            else:
                print(f"Nose model file not found at: {MODEL_PATHS['nose']}")
        except Exception as e:
            print(f"Error loading nose model: {str(e)}")
        
        try:
            # Lips model
            print(f"Loading lips model from: {MODEL_PATHS['lips']}")
            if os.path.exists(MODEL_PATHS['lips']):
                lips_model = LipsUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(device)
                lips_model.load_state_dict(torch.load(MODEL_PATHS['lips'], map_location=device))
                lips_model.eval()
                models['lips'] = lips_model
                models_loaded += 1
            else:
                print(f"Lips model file not found at: {MODEL_PATHS['lips']}")
        except Exception as e:
            print(f"Error loading lips model: {str(e)}")
        
        if models_loaded > 0:
            print(f"Successfully loaded {models_loaded} models")
            return True
        else:
            print("No models were loaded successfully")
            return False
    except Exception as e:
        print(f"Error in load_models: {str(e)}")
        return False

def preprocess_image(image_data, size=128):
    """Preprocess image for model input"""
    # Convert base64 to image
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        # Extract the base64 part
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    else:
        raise ValueError("Image data must be a base64 encoded string")
    
    # Resize image
    img_resized = img.resize((size, size))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    return img, img_tensor

def predict_mask(model, img_tensor):
    """Generate segmentation mask"""
    with torch.no_grad():
        output = model(img_tensor)
        predicted_mask = torch.sigmoid(output) > 0.5
    
    # Convert to numpy array
    predicted_mask = predicted_mask.cpu().squeeze().numpy().astype(np.uint8)
    
    return predicted_mask

def create_overlay_image(original_img, masks, selected_features):
    """Create an overlay image with colored masks"""
    # Convert PIL image to numpy array
    img_np = np.array(original_img)
    
    # Resize original image if needed
    if img_np.shape[0] > 800 or img_np.shape[1] > 800:
        aspect_ratio = img_np.shape[1] / img_np.shape[0]
        if img_np.shape[0] > img_np.shape[1]:
            new_height = 800
            new_width = int(aspect_ratio * 800)
        else:
            new_width = 800
            new_height = int(800 / aspect_ratio)
        img_np = cv2.resize(img_np, (new_width, new_height))
    
    h, w = img_np.shape[:2]
    
    # Create colored overlay
    colored_mask = np.zeros_like(img_np)
    
    for feature, mask in masks.items():
        if feature in selected_features and mask is not None:
            color = FEATURE_COLORS[feature]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            for c in range(3):
                colored_mask[:, :, c] += mask_resized * color[c]
    
    # Blend the original image with the colored mask
    alpha = 0.4
    blended = cv2.addWeighted(img_np, 1, colored_mask, alpha, 0)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', blended)
    blended_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return blended_b64

def create_mask_image(original_img, masks, selected_features):
    """Create a binary mask image with colored regions"""
    img_np = np.array(original_img)
    
    # Resize original image if needed
    if img_np.shape[0] > 800 or img_np.shape[1] > 800:
        aspect_ratio = img_np.shape[1] / img_np.shape[0]
        if img_np.shape[0] > img_np.shape[1]:
            new_height = 800
            new_width = int(aspect_ratio * 800)
        else:
            new_width = 800
            new_height = int(800 / aspect_ratio)
        img_np = cv2.resize(img_np, (new_width, new_height))
    
    h, w = img_np.shape[:2]
    
    # Create RGB mask
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for feature, mask in masks.items():
        if feature in selected_features and mask is not None:
            color = FEATURE_COLORS[feature]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            rgb_mask[mask_resized == 1] = color
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', rgb_mask)
    mask_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return mask_b64

def generate_placeholder_result(image_data, features, output_type):
    """Generate a placeholder result when no models are available"""
    # Decode the base64 image
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)
    
    # Resize if needed
    if img_np.shape[0] > 800 or img_np.shape[1] > 800:
        aspect_ratio = img_np.shape[1] / img_np.shape[0]
        if img_np.shape[0] > img_np.shape[1]:
            new_height = 800
            new_width = int(aspect_ratio * 800)
        else:
            new_width = 800
            new_height = int(800 / aspect_ratio)
        img_np = cv2.resize(img_np, (new_width, new_height))
    
    # Add text overlay explaining models are not available
    h, w = img_np.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_np, 'Model files not found', (int(w/4), int(h/2)), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_np, 'Please train models first', (int(w/4), int(h/2) + 40), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', img_np)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return result_b64

@app.route('/api/segment', methods=['POST'])
def segment_image():
    if not models:
        try:
            # Get request data for placeholder response
            data = request.json
            image_data = data.get('image')
            features = data.get('features', ['eye', 'nose', 'lips'])
            output_type = data.get('outputType', 'overlay')
            
            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400
                
            # Generate placeholder result
            result_image = generate_placeholder_result(image_data, features, output_type)
            
            response = {
                'segmented_image': f'data:image/png;base64,{result_image}',
                'features': features,
                'output_type': output_type,
                'warning': 'Models not loaded. Showing placeholder result.'
            }
            
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': f'Models not loaded and error generating placeholder: {str(e)}'}), 500
    
    try:
        # Get request data
        data = request.json
        image_data = data.get('image')
        features = data.get('features', ['eye', 'nose', 'lips'])
        output_type = data.get('outputType', 'overlay')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        original_img, img_tensor = preprocess_image(image_data)
        
        # Run segmentation for each selected feature
        masks = {}
        for feature in ['eye', 'nose', 'lips']:
            if feature in features and feature in models:
                masks[feature] = predict_mask(models[feature], img_tensor)
        
        # Create output image based on selected type
        if output_type == 'overlay':
            result_image = create_overlay_image(original_img, masks, features)
        else:  # mask
            result_image = create_mask_image(original_img, masks, features)
        
        # Create response with image data
        response = {
            'segmented_image': f'data:image/png;base64,{result_image}',
            'features': features,
            'output_type': output_type
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in segment_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    model_status = {name: name in models for name in ['eye', 'nose', 'lips']}
    return jsonify({
        'status': 'running',
        'models': model_status,
        'device': str(device),
        'model_paths': {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}
    })

if __name__ == '__main__':
    print(f"Loading models on device: {device}")
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True) 