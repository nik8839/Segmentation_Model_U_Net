# Eye Segmentation from Human Faces

This project provides scripts to train a deep learning model for segmenting eyes from human faces using the CelebAMask-HQ dataset. The model is built using a lightweight U-Net architecture optimized for the specific task of eye segmentation.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The project uses the CelebAMask-HQ dataset, which has the following structure:

```
CelebAMask-HQ/
├── CelebA-HQ-img/             # Directory containing original face images (*.jpg)
└── CelebAMask-HQ-mask-anno/   # Directory containing mask annotations
    ├── 0/                     # Subdirectories containing mask annotations
    ├── 1/
    └── ...
```

Each mask annotation subdirectory contains various facial component masks, including:

- `*_l_eye.png`: Left eye mask
- `*_r_eye.png`: Right eye mask

## Training the Eye Segmentation Model

To train the eye segmentation model, run:

```bash
python eye_segmentation.py
```

This script will:

1. Load the CelebAMask-HQ dataset
2. Find and process images with eye masks
3. Train a U-Net model for binary eye segmentation
4. Save the model checkpoints and final model in the `output` directory
5. Generate training history plots and sample predictions

### Training Parameters

You can modify the following parameters in `eye_segmentation.py`:

- `BATCH_SIZE`: Batch size for training (default: 16)
- `LEARNING_RATE`: Learning rate for optimizer (default: 0.001)
- `EPOCHS`: Number of training epochs (default: 25)
- `IMG_SIZE`: Size to resize images (default: 256)

## Inference

To run inference on new images using the trained model:

```bash
python inference.py --image path/to/image.jpg
```

### Inference Options

- `--image`: Path to input image (required)
- `--model`: Path to trained model (default: 'output/eye_segmentation_model.pth')
- `--output`: Path to save output image (default: 'eye_segmentation_result.png')
- `--device`: Device to use for inference ('cuda' or 'cpu')
- `--overlay`: Add this flag to show only the original and overlayed images

## Model Architecture

The model uses a lightweight U-Net architecture with the following components:

1. **Encoder**: Multiple convolutional blocks that progressively downsample the input image.
2. **Decoder**: Upsampling blocks that restore the spatial dimensions while using skip connections from the encoder.
3. **Output Layer**: A 1x1 convolution that produces a binary segmentation mask for eyes.

## Results

After training, you can find the following in the `output` directory:

- `eye_segmentation_model.pth`: The final trained model
- `model_checkpoint_epoch_*.pth`: Model checkpoints at each epoch
- `training_history.png`: Plot of training and validation losses
- `sample_predictions.png`: Visualization of model predictions on sample images

## Example Use Cases

1. **AR Filters**: Use eye segmentation for precise placement of virtual glasses or eye effects
2. **Gaze Tracking**: As a preprocessing step for gaze tracking applications
3. **Eye Analysis**: For medical applications or biometric authentication
4. **Eye Enhancement**: For photo editing applications

## Limitations

- The model is trained on the CelebAMask-HQ dataset, which may not generalize to all face types
- Performance may vary with different lighting conditions or occlusions (like glasses)
- The model is optimized for frontal face images
