import os
import glob
import shutil
import random

# Set dataset paths
data_dir = "data"  # Change this to your dataset path
images_dir = os.path.join(data_dir, "converted")
labels_dir = os.path.join(data_dir, "labels")
output_dir = os.path.join(data_dir, "split_dataset")

# Create train/val folders
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Get image files
image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
random.shuffle(image_files)

# Split ratio (80% train, 20% val)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_files(files, subset):
    for img_path in files:
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))
        
        # Move image
        shutil.copy(img_path, os.path.join(output_dir, f"images/{subset}", filename))
        
        # Move label if exists
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_dir, f"labels/{subset}", filename.replace(".jpg", ".txt")))

# Move files to respective folders
move_files(train_files, "train")
move_files(val_files, "val")

print("Dataset split complete. Check the 'split_dataset' folder.")
