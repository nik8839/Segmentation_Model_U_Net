import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Import model architecture from eye and nose segmentation scripts
from eye_segmentation import UNet as EyeUNet
from nose_segmentation import UNet as NoseUNet

def parse_args():
    parser = argparse.ArgumentParser(description='Combined Eye-Nose Segmentation Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--eye_model', type=str, default='output/eye_segmentation_model.pth', 
                        help='Path to trained eye segmentation model')
    parser.add_argument('--nose_model', type=str, default='output_nose/nose_segmentation_model.pth', 
                        help='Path to trained nose segmentation model')
    parser.add_argument('--output', type=str, default='combined_eye_nose_result.png', 
                        help='Path to save output image')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    parser.add_argument('--overlay', action='store_true', help='Overlay mask on original image')
    return parser.parse_args()

def preprocess_image(image_path, size=128):
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize image
    img_resized = img.resize((size, size))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)  # Add batch dimension
    
    return img, img_tensor

def predict_mask(model, img_tensor, device):
    # Set model to evaluation mode
    model.eval()
    
    # Move tensor to the right device
    img_tensor = img_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        predicted_mask = torch.sigmoid(output) > 0.5
    
    # Convert to numpy array
    predicted_mask = predicted_mask.cpu().squeeze().numpy().astype(np.uint8)
    
    return predicted_mask

def overlay_masks(img, eye_mask, nose_mask, alpha=0.4):
    # Convert PIL image to numpy array
    img_np = np.array(img)
    
    # Resize masks to match image dimensions
    h, w = img_np.shape[:2]
    eye_mask_resized = cv2.resize(eye_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    nose_mask_resized = cv2.resize(nose_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create colored masks (blue for eyes, orange for nose)
    eye_color = (0, 0, 255)  # Blue in BGR
    nose_color = (255, 165, 0)  # Orange in BGR
    
    # Create a colored mask
    colored_mask = np.zeros_like(img_np)
    
    # Apply colors to the respective masks
    for c in range(3):
        colored_mask[:, :, c] += eye_mask_resized * eye_color[c]
        colored_mask[:, :, c] += nose_mask_resized * nose_color[c]
    
    # Blend the original image with the colored mask
    blended = cv2.addWeighted(img_np, 1, colored_mask, alpha, 0)
    
    return blended

def save_results(original_img, eye_mask, nose_mask, combined_mask, output_path, overlay=False):
    if overlay:
        plt.figure(figsize=(12, 6))
        
        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Display overlayed image
        overlayed = overlay_masks(original_img, eye_mask, nose_mask)
        plt.subplot(1, 2, 2)
        plt.imshow(overlayed)
        plt.title('Eye-Nose Segmentation')
        plt.axis('off')
    else:
        plt.figure(figsize=(20, 10))
        
        # Display original image
        plt.subplot(2, 3, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Display eye mask
        plt.subplot(2, 3, 2)
        plt.imshow(eye_mask, cmap='Blues')
        plt.title('Eye Mask')
        plt.axis('off')
        
        # Display nose mask
        plt.subplot(2, 3, 3)
        plt.imshow(nose_mask, cmap='Oranges')
        plt.title('Nose Mask')
        plt.axis('off')
        
        # Display combined binary mask
        plt.subplot(2, 3, 4)
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Combined Mask')
        plt.axis('off')
        
        # Display overlayed image
        overlayed = overlay_masks(original_img, eye_mask, nose_mask)
        plt.subplot(2, 3, 5)
        plt.imshow(overlayed)
        plt.title('Overlayed Result')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load eye model
    eye_model = EyeUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(device)
    try:
        eye_model.load_state_dict(torch.load(args.eye_model, map_location=device))
        print(f"Eye model loaded from {args.eye_model}")
    except Exception as e:
        print(f"Error loading eye model: {e}")
        return
    
    # Load nose model
    nose_model = NoseUNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(device)
    try:
        nose_model.load_state_dict(torch.load(args.nose_model, map_location=device))
        print(f"Nose model loaded from {args.nose_model}")
    except Exception as e:
        print(f"Error loading nose model: {e}")
        return
    
    # Preprocess image
    original_img, img_tensor = preprocess_image(args.image)
    
    # Predict masks
    eye_mask = predict_mask(eye_model, img_tensor, device)
    nose_mask = predict_mask(nose_model, img_tensor, device)
    
    # Combine masks
    combined_mask = np.maximum(eye_mask, nose_mask)
    
    # Save results
    save_results(original_img, eye_mask, nose_mask, combined_mask, args.output, args.overlay)

if __name__ == "__main__":
    main() 