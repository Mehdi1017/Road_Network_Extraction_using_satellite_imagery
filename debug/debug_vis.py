import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import RoadDataset

# Load your dataset
dataset = RoadDataset(data_root='/home/mehdi/thesis/AOI_2_Vegas')

# Get one sample
image, mask = dataset[0] # Try index 0, 10, 50

# Convert back to numpy for plotting
# Image is (3, H, W) -> (H, W, 3)
img_np = image.permute(1, 2, 0).numpy()
# Mask is (1, H, W) -> (H, W)
mask_np = mask.squeeze().numpy()

print(f"Image Max Value: {img_np.max()}")
print(f"Mask Unique Values: {np.unique(mask_np)}")

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title("Satellite Image")

plt.subplot(1, 2, 2)
plt.imshow(mask_np, cmap='gray')
plt.title("Ground Truth Mask")

plt.show()
plt.savefig("debug_preview.png")