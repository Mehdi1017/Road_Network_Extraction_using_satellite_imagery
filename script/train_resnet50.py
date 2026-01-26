import torch
import albumentations as A
import segmentation_models_pytorch as smp
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
from initial.dataset import RoadDataset
from model import get_model

# --- THESIS CONFIGURATION ---
ARCHITECTURE = 'unet'         
ENCODER = 'resnet50'            # Options: 'resnet50', 'mit_b3'

DATA_DIR = '../src/AOI_2_Vegas' 
EPOCHS = 100                  # Increased max epochs because we have Early Stopping
BATCH_SIZE = 4
PATIENCE = 30                 # Stop if no improvement for 10 epochs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. DATA TRANSFORMS ---
train_transform = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=0), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
])

# --- 2. DATASETS ---
train_dataset = RoadDataset(file_list_path="../src/train_list.txt", transform=train_transform)
val_dataset = RoadDataset(file_list_path="../src/val_list.txt", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

# --- 3. MODEL SETUP ---
print(f"Initializing {ARCHITECTURE} with {ENCODER} backbone...")
model = get_model(ARCHITECTURE, ENCODER).to(DEVICE)

# --- 4. SMART CONFIGURATION ---
if 'mit' in ENCODER:
    print(">>> MODE: TRANSFORMER BACKBONE DETECTED")
    print(" - Optimizer: AdamW")
    print(" - Loss: BCEWithLogitsLoss")
    print(" - AMP: DISABLED")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-2)
    loss_fn_internal = torch.nn.BCEWithLogitsLoss() 
    use_amp = True 
    
    def loss_fn(logits, targets):
        return loss_fn_internal(logits, targets)

else:
    print(">>> MODE: CNN BACKBONE DETECTED")
    print(" - Optimizer: Adam")
    print(" - Loss: Dice + Focal")
    print(" - AMP: ENABLED")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3.54e-4, weight_decay=1.8e-5)
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='binary')
    use_amp = True
    
    def loss_fn(logits, targets):
        d_loss = dice_loss(logits, targets)
        probs = logits.sigmoid()
        f_loss = focal_loss(probs, targets)
        return (0.9 * f_loss) + (0.1 * d_loss)

# --- 5. SCHEDULER ---
total_steps = EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=3.54e-4, 
    total_steps=total_steps, 
    pct_start=0.1, 
    div_factor=10, 
    final_div_factor=100
)

scaler = torch.amp.GradScaler('cuda') 

def compute_iou(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    masks = masks.float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# --- 6. TRAINING LOOP (With Early Stopping) ---
print(f"Starting training... (AMP: {use_amp}, Early Stopping Patience: {PATIENCE})")

best_iou = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        # Forward
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = loss_fn(logits, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- SAFETY FIX: GRADIENT CLIPPING ---
            scaler.unscale_(optimizer) # Important: Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # -------------------------------------
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            optimizer.zero_grad()
            loss.backward()

            # --- SAFETY FIX ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ------------------

            optimizer.step()
        
        # Debug Print
        if i == 0:
            probs = logits.sigmoid()
            p_mean = probs.mean().item()
            t_mean = masks.float().mean().item()
            lr = scheduler.get_last_lr()[0]
            print(f"\n[Epoch {epoch+1}] Pred Mean: {p_mean:.4f} (Target: {t_mean:.4f}) | LR: {lr:.2e}")

        scheduler.step()
        epoch_loss += loss.item()
        
    avg_train_loss = epoch_loss / len(train_loader)
    
    # --- VALIDATION ---
    model.eval()
    val_iou_score = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    preds = logits.sigmoid()
            else:
                logits = model(images)
                preds = logits.sigmoid()
            
            val_iou_score += compute_iou(preds, masks)
            
    avg_val_iou = val_iou_score / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")
    
    # --- EARLY STOPPING LOGIC ---
    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        patience_counter = 0  # Reset counter
        torch.save(model.state_dict(), f'{ARCHITECTURE}_{ENCODER}_best.pth')
        print(f"  --> New Best Model Saved! ({best_iou:.4f})")
    else:
        patience_counter += 1
        print(f"  ... No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n🛑 EARLY STOPPING TRIGGERED at Epoch {epoch+1}")
            print(f"Best Validation IoU: {best_iou:.4f}")
            break

print("Training finished.")
torch.save(model.state_dict(), f'{ARCHITECTURE}_{ENCODER}_final.pth')