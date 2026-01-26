import torch
import albumentations as A
import segmentation_models_pytorch as smp
import numpy as np
import os
import sys
from scipy.optimize import differential_evolution
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from initial.dataset import RoadDataset
from model import get_model

# --- CONFIGURATION ---
DATA_DIR = '/home/mehdi/thesis/src/AOI_2_Vegas' 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4 
SEARCH_EPOCHS = 5  # ResNet learns fast, 5 epochs is enough to spot the trend

# --- DATA PREP ---
# We use the same normalization as training to ensure valid results
train_transform = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=0), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
])

print("Loading dataset for ResNet tuning...")
# Robust Dataset Loading (ignores XMLs automatically via the updated dataset.py)
full_train_dataset = RoadDataset(data_root=DATA_DIR, transform=train_transform)
full_val_dataset = RoadDataset(data_root=DATA_DIR, transform=val_transform)

train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

# Validation subset (300 images) for speed
val_indices = torch.randperm(len(full_val_dataset))[:300].tolist()
val_dataset = Subset(full_val_dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# --- OBJECTIVE FUNCTION ---
def objective_function(params):
    """
    Minimizes Validation Loss.
    params[0]: Learning Rate (Log scale)
    params[1]: Weight Decay (Log scale)
    params[2]: Focal Loss Ratio (Linear 0.1 - 0.9)
    """
    # Decode parameters
    lr = 10 ** params[0]
    wd = 10 ** params[1]
    focal_ratio = params[2]
    dice_ratio = 1.0 - focal_ratio
    
    print(f"\n🧪 Testing: LR={lr:.2e} | WD={wd:.2e} | Loss: {focal_ratio:.2f}*Focal + {dice_ratio:.2f}*Dice")
    
    # 1. Initialize Model (ResNet50)
    model = get_model('unet', 'resnet50').to(DEVICE)
    
    # 2. Optimizer (Standard Adam for ResNet)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # 3. Loss (Dynamic Combination)
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='binary')
    
    def loss_fn(logits, targets):
        d_loss = dice_loss(logits, targets)
        probs = logits.sigmoid()
        f_loss = focal_loss(probs, targets)
        return (focal_ratio * f_loss) + (dice_ratio * d_loss)
    
    # 4. Scheduler (Standard OneCycle)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, 
        total_steps=SEARCH_EPOCHS * len(train_loader),
        pct_start=0.3
    )
    
    # 5. Training Loop
    model.train()
    best_loss_in_run = 10.0
    
    try:
        for epoch in range(SEARCH_EPOCHS):
            # Train
            for images, masks in train_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                
                optimizer.zero_grad()
                torch.cuda.amp.GradScaler().scale(loss).backward()
                optimizer.step()
                scheduler.step()
                
            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    
                    with torch.amp.autocast('cuda'):
                        logits = model(images)
                        loss = loss_fn(logits, masks)
                        
                    val_loss_accum += loss.item()
            
            avg_val_loss = val_loss_accum / len(val_loader)
            model.train()
            
            print(f"   [Ep {epoch+1}] Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_loss_in_run:
                best_loss_in_run = avg_val_loss
            
            if np.isnan(avg_val_loss) or avg_val_loss > 2.0:
                return 10.0 # Fail early
                
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 10.0 

    print(f"  -> Final Score: {best_loss_in_run:.4f}")
    return best_loss_in_run

# --- RUN DIFFERENTIAL EVOLUTION ---
def run_tuning():
    print("--- STARTING RESNET50 HYPERPARAMETER SEARCH ---")
    
    # Bounds:
    # 1. LR: 1e-5 to 1e-3 (Log scale -5 to -3)
    # 2. WD: 1e-6 to 1e-2 (Log scale -6 to -2)
    # 3. Focal Ratio: 0.1 to 0.9 (Linear)
    bounds = [
        (-5.0, -3.0), 
        (-6.0, -2.0),
        (0.1, 0.9)
    ]
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=5,    # 5 Generations
        popsize=4,    # Population 12 (4*3 params)
        tol=0.01,
        disp=True,
        workers=1     
    )
    
    best_lr = 10 ** result.x[0]
    best_wd = 10 ** result.x[1]
    best_focal = result.x[2]
    best_dice = 1.0 - best_focal
    
    print("\n🏆 OPTIMIZATION COMPLETE!")
    print(f"Best Val Loss: {result.fun:.4f}")
    print(f"Optimal Learning Rate: {best_lr:.2e}")
    print(f"Optimal Weight Decay:  {best_wd:.2e}")
    print(f"Optimal Loss Balance:  {best_focal:.2f} * Focal + {best_dice:.2f} * Dice")

if __name__ == "__main__":
    run_tuning()