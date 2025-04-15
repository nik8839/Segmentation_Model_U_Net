import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Import model architecture from nose_segmentation.py
from nose_segmentation import UNet

def parse_args():
    parser = argparse.ArgumentParser(description='Nose Segmentation Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='output_nose/nose_segmentation_model.pth', 
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='nose_segmentation_result.png', 
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

def overlay_mask(img, mask, alpha=0.4, color=(255, 165, 0)):  # Orange color for nose
    # Convert PIL image to numpy array
    img_np = np.array(img)
    
    # Resize mask to match image dimensions
    mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create a colored mask
    colored_mask = np.zeros_like(img_np)
    for c in range(3):
        colored_mask[:, :, c] = mask_resized * color[c]
    
    # Blend the original image with the colored mask
    blended = cv2.addWeighted(img_np, 1, colored_mask, alpha, 0)
    
    return blended

def save_results(original_img, predicted_mask, output_path, overlay=False):
    # Create figure with subplots
    if overlay:
        plt.figure(figsize=(12, 6))
        
        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Display overlayed image
        overlayed = overlay_mask(original_img, predicted_mask)
        plt.subplot(1, 2, 2)
        plt.imshow(overlayed)
        plt.title('Nose Segmentation Result')
        plt.axis('off')
    else:
        plt.figure(figsize=(18, 6))
        
        # Display original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Display predicted mask
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Nose Segmentation Mask')
        plt.axis('off')
        
        # Display overlayed image
        overlayed = overlay_mask(original_img, predicted_mask)
        plt.subplot(1, 3, 3)
        plt.imshow(overlayed)
        plt.title('Overlayed Result')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load model
    device = torch.device(args.device)
    model = UNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Preprocess image
    original_img, img_tensor = preprocess_image(args.image)
    
    # Make prediction
    predicted_mask = predict_mask(model, img_tensor, device)
    
    # Save results
    save_results(original_img, predicted_mask, args.output, args.overlay)

if __name__ == "__main__":
    main() 