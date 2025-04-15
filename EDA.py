import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set the dataset directory
dataset_dir = './CelebAMask-HQ'

# Check the structure of the dataset
print("Dataset Directory Structure:")
print(f"Dataset directory: {dataset_dir}")
print(f"Contains the following files and directories:")
for root, dirs, files in os.walk(dataset_dir):
    print(f"Root: {root}, Dirs: {len(dirs)}, Files: {len(files)}")

# Example: Load attribute file if available
attributes_file = os.path.join(dataset_dir, 'list_attr_celeba.txt')
if os.path.exists(attributes_file):
    # Load attributes
    with open(attributes_file, 'r') as f:
        lines = f.readlines()
    columns = lines[1].strip().split()
    data = [line.strip().split() for line in lines[2:]]
    df = pd.DataFrame(data, columns=['Image_ID'] + columns)
    df.set_index('Image_ID', inplace=True)
    print("Attributes DataFrame:")
    print(df.head())

    # Convert attributes to numeric
    df = df.apply(pd.to_numeric)

    # Plot attribute distribution
    plt.figure(figsize=(12, 6))
    df.sum().sort_values(ascending=False).plot(kind='bar')
    plt.title('Attribute Distribution')
    plt.xlabel('Attributes')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

# Example: Visualize some images
image_dir = os.path.join(dataset_dir, 'images')
if os.path.exists(image_dir):
    sample_images = os.listdir(image_dir)[:5]
    plt.figure(figsize=(15, 5))
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(img_name)
    plt.show()
else:
    print("Image directory not found.")