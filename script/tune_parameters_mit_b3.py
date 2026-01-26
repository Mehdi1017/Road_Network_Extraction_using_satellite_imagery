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
BATCH_SIZE = 2

# REDUCED: 5 epochs is enough to spot a good learning curve vs a bad one
SEARCH_EPOCHS = 5  

# --- DATA PREP ---
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

print("Loading dataset for tuning...")
# Note: Using the full dataset. Since search_epochs is low, this is fine.
full_train_dataset = RoadDataset(data_root=DATA_DIR, transform=train_transform)
full_val_dataset = RoadDataset(data_root=DATA_DIR, transform=val_transform)

train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# Validation subset (300 images) to speed up the evaluation step
val_indices = torch.randperm(len(full_val_dataset))[:300].tolist()
val_dataset = Subset(full_val_dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# --- OBJECTIVE FUNCTION ---
def objective_function(params):
    """
    Minimizes Validation Loss based on 4 parameters.
    """
    # Decode parameters
    lr = 10 ** params[0]
    weight_decay = 10 ** params[1]
    pct_start = params[2]
    pos_weight_val = params[3]
    
    print(f"\n🧪 Testing: LR={lr:.2e} | WD={weight_decay:.2e} | Warmup={pct_start:.2f} | PosW={pos_weight_val:.2f}")
    
    # 1. Initialize Model (Fresh)
    model = get_model('unet', 'mit_b3').to(DEVICE)
    
    # 2. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 3. Loss
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # 4. Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, 
        total_steps=SEARCH_EPOCHS * len(train_loader),
        pct_start=pct_start, 
        div_factor=10,
        final_div_factor=100
    )
    
    # 5. Training Loop
    model.train()
    best_loss_in_run = 10.0
    
    try:
        for epoch in range(SEARCH_EPOCHS):
            # Train
            for images, masks in train_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                # FP32 for stability during search
                with torch.amp.autocast('cuda', enabled=False):
                    logits = model(images)
                    loss = criterion(logits, masks)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    logits = model(images)
                    val_loss_accum += criterion(logits, masks).item()
            
            avg_val_loss = val_loss_accum / len(val_loader)
            model.train()
            
            print(f"   [Ep {epoch+1}] Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_loss_in_run:
                best_loss_in_run = avg_val_loss
            
            # Pruning: If loss explodes (>1.5) or Nan, kill this run
            if avg_val_loss > 1.5 or np.isnan(avg_val_loss):
                return 10.0 
                
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 10.0 

    print(f"  -> Final Score: {best_loss_in_run:.4f}")
    return best_loss_in_run

# --- RUN DIFFERENTIAL EVOLUTION ---
def run_tuning():
    print("--- STARTING OVERNIGHT HYPERPARAMETER SEARCH (Approx 6 Hours) ---")
    
    # Bounds:
    # LR: 1e-5 to 1e-3
    # WD: 1e-6 to 1e-2
    # Warmup: 10% to 50%
    # PosWeight: 1.0 to 15.0
    bounds = [
        (-5.0, -3.0), 
        (-6.0, -2.0),
        (0.1, 0.5),
        (1.0, 15.0)
    ]
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=5,    # Reduced to 5 generations
        popsize=3,    # Reduced population to 12 (3*4)
        tol=0.01,
        disp=True,
        workers=1     
    )
    
    best_lr = 10 ** result.x[0]
    best_wd = 10 ** result.x[1]
    best_warmup = result.x[2]
    best_pos_w = result.x[3]
    
    print("\n🏆 OPTIMIZATION COMPLETE!")
    print(f"Best Val Loss: {result.fun:.4f}")
    print(f"Optimal Learning Rate: {best_lr:.2e}")
    print(f"Optimal Weight Decay:  {best_wd:.2e}")
    print(f"Optimal Warmup (pct):  {best_warmup:.2f}")
    print(f"Optimal Pos Weight:    {best_pos_w:.2f}")

if __name__ == "__main__":
    run_tuning()