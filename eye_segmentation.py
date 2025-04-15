import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.002
EPOCHS = 10
IMG_SIZE = 128
DATASET_LIMIT = 2000

# Paths
MASK_ROOT = 'CelebAMask-HQ/CelebAMask-HQ-mask-anno'
IMG_ROOT = 'CelebAMask-HQ/CelebA-HQ-img'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a lightweight U-Net model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, depth=3, base_filters=16):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(depth):
            in_channels = n_channels if i == 0 else base_filters * (2 ** (i-1))
            out_channels = base_filters * (2 ** i)
            self.encoders.append(DoubleConv(in_channels, out_channels))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(depth - 1):
            in_channels = base_filters * (2 ** (depth - 1 - i))
            out_channels = base_filters * (2 ** (depth - 2 - i))
            self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(in_channels, out_channels))
        
        # Final layer
        self.final_conv = nn.Conv2d(base_filters, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        encoder_features = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            if i < self.depth - 1:
                encoder_features.append(x)
                x = self.pools[i](x)
        
        # Decoder path
        for i in range(self.depth - 1):
            x = self.upconvs[i](x)
            x_enc = encoder_features[-(i+1)]
            
            # Handle case when dimensions don't match
            if x.size() != x_enc.size():
                diff_h = x_enc.size(2) - x.size(2)
                diff_w = x_enc.size(3) - x.size(3)
                x = nn.functional.pad(x, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
            
            x = torch.cat([x, x_enc], dim=1)
            x = self.decoders[i](x)
        
        # Final layer
        x = self.final_conv(x)
        return x

# Dataset class to load images and eye masks
class EyeSegmentationDataset(Dataset):
    def __init__(self, image_ids, transform=None):
        self.image_ids = image_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(IMG_ROOT, f"{img_id}.jpg")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Create empty binary mask for eyes
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        # Find and merge left eye and right eye masks
        for eye_type in ['l_eye', 'r_eye']:
            mask_path_pattern = os.path.join(MASK_ROOT, f"*/{img_id}_{eye_type}.png")
            eye_mask_paths = glob.glob(mask_path_pattern)
            
            if eye_mask_paths:
                eye_mask_path = eye_mask_paths[0]
                eye_mask = np.array(Image.open(eye_mask_path).convert('L'))
                eye_mask = cv2.resize(eye_mask, (IMG_SIZE, IMG_SIZE))
                eye_mask = (eye_mask > 0).astype(np.uint8)
                mask = np.maximum(mask, eye_mask)
        
        # Convert image and mask to tensors
        img = transforms.Resize((IMG_SIZE, IMG_SIZE))(img)
        
        if self.transform:
            img = self.transform(img)
        
        # Convert mask to tensor (add a channel dimension)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return img, mask

# Function to find all valid image IDs
def get_valid_image_ids():
    all_image_files = glob.glob(os.path.join(IMG_ROOT, "*.jpg"))
    image_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_image_files]
    
    # Filter to only include images that have eye masks
    valid_ids = []
    for img_id in tqdm(image_ids, desc="Checking for eye masks"):
        l_eye_exists = len(glob.glob(os.path.join(MASK_ROOT, f"*/{img_id}_l_eye.png"))) > 0
        r_eye_exists = len(glob.glob(os.path.join(MASK_ROOT, f"*/{img_id}_r_eye.png"))) > 0
        
        if l_eye_exists or r_eye_exists:
            valid_ids.append(img_id)
            
            # Limit dataset size for faster training
            if len(valid_ids) >= DATASET_LIMIT:
                break
    
    print(f"Found {len(valid_ids)} images with eye masks (limited to {DATASET_LIMIT})")
    return valid_ids

# Training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    return running_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

# Plot training history
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

# Plot model predictions on sample images
def plot_predictions(model, val_loader, num_samples=5):
    model.eval()
    images, masks = next(iter(val_loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].numpy()
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        preds = preds.cpu().numpy()
    
    images = images.cpu().numpy()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Display original image
        img = np.transpose(images[i], (1, 2, 0))
        img = (img * 0.5) + 0.5  # Unnormalize
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # Display ground truth mask
        axes[i, 1].imshow(masks[i, 0], cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Display predicted mask
        axes[i, 2].imshow(preds[i, 0], cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'))
    plt.close()

def main():
    # Get valid image IDs
    valid_ids = get_valid_image_ids()
    
    # Split into train and validation sets
    train_ids, val_ids = train_test_split(valid_ids, test_size=0.2, random_state=42)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = EyeSegmentationDataset(train_ids, transform=transform)
    val_dataset = EyeSegmentationDataset(val_ids, transform=transform)
    
    # Create data loaders with fewer workers
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model with smaller architecture
    model = UNet(n_channels=3, n_classes=1, depth=3, base_filters=16).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        val_loss = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(OUTPUT_DIR, f'model_checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'eye_segmentation_model.pth'))
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Plot sample predictions
    plot_predictions(model, val_loader)
    
    print("Training complete! Model saved to:", os.path.join(OUTPUT_DIR, 'eye_segmentation_model.pth'))

if __name__ == "__main__":
    main() 