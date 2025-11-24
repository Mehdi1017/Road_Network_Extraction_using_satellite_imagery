import torch
import albumentations as A
import segmentation_models_pytorch as smp
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import RoadDataset
from model import get_model

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = '/home/mehdi/thesis/AOI_2_Vegas' 

# --- 1. TRANSFORMS ---
# CenterCrop + Normalize (Corrected for 0-1 inputs)
overfit_transform = A.Compose([
    A.CenterCrop(height=512, width=512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225), 
        max_pixel_value=1.0
    ),
])

# --- 2. DATASET ---
print(f"Loading dataset from {DATA_DIR}...")
ds = RoadDataset(data_root=DATA_DIR, transform=overfit_transform)

if len(ds) == 0:
    print("❌ ERROR: No images found! Check DATA_DIR path.")
    exit()

# Take the first image and repeat it 4 times
subset = Subset(ds, indices=[0, 0, 0, 0]) 
loader = DataLoader(subset, batch_size=4, shuffle=False)

# --- 3. CONTROL GROUP MODEL (ResNet18) ---
# We switch to ResNet18 because it is extremely robust.
# If this learns, your data is fine. If this fails, your data is broken.
print("Initializing U-Net ResNet18 (Control Group)...")
model = get_model('unet', 'resnet18').to(DEVICE)

# --- 4. SETTINGS ---
# Standard settings (High LR is fine for ResNet)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = torch.nn.BCEWithLogitsLoss() 
scaler = torch.amp.GradScaler('cuda')

print("--- STARTING CONTROL GROUP OVERFIT TEST ---")
print("Goal: Loss should drop to < 0.01")

for epoch in range(1, 31):
    model.train()
    epoch_loss = 0
    
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        # --- INPUT DEBUGGING ---
        if epoch == 1 and i == 0:
            print(f"[DEBUG] Input Stats: Min={images.min():.3f}, Max={images.max():.3f}, Mean={images.mean():.3f}")
            print(f"[DEBUG] Target Mean: {masks.float().mean().item():.4f}")
        # -----------------------
        
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, masks)
            
            # Monitor Probability
            probs = logits.sigmoid()
            pred_mean = probs.mean().item()
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Pred Mean: {pred_mean:.4f}")
    
    # Early success check
    if epoch_loss < 0.05:
        print("✅ SUCCESS! ResNet18 memorized the image. Data pipeline is VALID.")
        break