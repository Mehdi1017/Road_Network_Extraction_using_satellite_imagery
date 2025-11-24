import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import glob

# Load a prediction
files = glob.glob("test_results/predictions/*.png")
img = np.array(Image.open(files[0])) > 127 # Load first image

# Skeletonize
skel = skeletonize(img)

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Prediction Mask")

plt.subplot(1, 2, 2)
plt.imshow(skel, cmap='gray')
plt.title("Skeleton (The Graph Basis)")
plt.show()
plt.savefig("skeleton_debug.png")