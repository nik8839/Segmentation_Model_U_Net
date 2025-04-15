# Facial Segmentation with CelebAMask-HQ

This project implements a lightweight facial segmentation model that runs on CPU. It uses the CelebAMask-HQ dataset to train a model for segmenting facial features, specifically targeting eyes, eyebrows, nose, mouth, and lips.

## Dataset

The CelebAMask-HQ dataset includes:

- High-quality face images (CelebA-HQ-img)
- Mask annotations for 19 facial components (CelebAMask-HQ-mask-anno)

## Features

- Lightweight segmentation model designed to run on CPU
- Trains on the CelebAMask-HQ dataset
- Segments facial features with different color codes
- Provides visualization of segmentation results

## Model Architecture

The project includes two model architectures:

1. **LightweightUNet**: A simplified U-Net architecture with reduced parameters
2. **LightweightSegmentationModel**: An even more efficient model using depthwise separable convolutions

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the facial segmentation model:

```bash
python face_segmentation.py
```

This will:

1. Load the CelebAMask-HQ dataset
2. Preprocess the images and mask annotations
3. Train the lightweight segmentation model
4. Save the trained model to `face_segmentation_model.pth`
5. Evaluate the model and visualize results

### Inference

To segment a face image using the trained model:

```bash
python inference.py --image [path_to_image] --output [output_filename.png]
```

Arguments:

- `--image`: Path to the input face image (required)
- `--model`: Path to the trained model file (default: face_segmentation_model.pth)
- `--output`: Path to save the output image (default: segmentation_result.png)

## Color Coding

The segmentation results are color-coded as follows:

- Background: Black
- Eyes: Red
- Eyebrows: Green
- Nose: Blue
- Mouth: Yellow
- Lips: Magenta

## Implementation Details

- The model is implemented in PyTorch
- Input images are resized to 256x256 pixels
- Mask annotations for different parts are combined into a single segmentation mask
- The implementation uses depthwise separable convolutions for efficiency

## Results

After training, the model produces:

- Segmentation masks where each pixel is labeled with its corresponding facial part
- Visualization with color-coded facial features
- Training loss curve saved as `training_loss.png`
