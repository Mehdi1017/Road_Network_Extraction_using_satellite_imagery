import os
import glob
import random

# Define your data root
DATA_ROOT = "../../src/AOI_2_Vegas" # Update when you add more cities

# Find all available images (PS-RGB)
all_images = glob.glob(os.path.join(DATA_ROOT, "PS-RGB", "*.tif"))
all_images.sort() # Sort to ensure reproducibility

# Shuffle with a fixed seed
random.seed(42)
random.shuffle(all_images)

# Define split ratios (e.g., 80% Train, 10% Val, 10% Test)
total = len(all_images)
train_split = int(0.8 * total)
val_split = int(0.1 * total)

train_files = all_images[:train_split]
val_files = all_images[train_split : train_split + val_split]
test_files = all_images[train_split + val_split:]

print(f"Total Images: {total}")
print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")

# Save lists to text files
with open("train_list.txt", "w") as f:
    f.write("\n".join(train_files))

with open("val_list.txt", "w") as f:
    f.write("\n".join(val_files))
    
with open("test_list.txt", "w") as f:
    f.write("\n".join(test_files))

print("Split lists saved successfully.")