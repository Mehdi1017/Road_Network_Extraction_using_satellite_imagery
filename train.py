import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# --- IMPORTS FROM SRC MODULE ---
from src.data_prep.dataset import RoadDataset
from src.models.architectures import get_model # unet getter
from src.models.modules import DeepLabV3PlusD3S2PP # custom module
from src.losses.topology_loss import TopologyAwareLoss

def parse_args():
    parser = argparse.ArgumentParser(description="DeepRoad: Train Network Architectures")
    
    # Core Setup
    parser.add_argument("--model", type=str, required=True, choices=["mit_b3", "resnet50", "d3s2pp"],
                        help="Architecture to train")
    parser.add_argument("--loss", type=str, default="pixel", choices=["pixel", "topo"],
                        help="Use 'pixel' (standard) or 'topo' (TopologyAwareLoss)")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--warmup", type=int, default=5, help="Epochs before Topo loss activates")
    
    # Checkpointing
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- 1. FILE PATHS ---
    PREFIX = f"{args.model}_{args.loss}"
    LOG_FILE = f"logs/training_log_{PREFIX}.csv"
    BEST_MODEL_PATH = f"weights/{PREFIX}_best.pth"
    LATEST_CHECKPOINT = f"weights/{PREFIX}_latest.pth"

    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'lr'])

    # --- 2. DATA LOADERS ---
    print(f"Loading datasets (Batch Size: {args.batch_size})...")
    train_transform = A.Compose([
        A.RandomCrop(512, 512),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_transform = A.Compose([
        A.PadIfNeeded(1312, 1312, border_mode=0), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = RoadDataset(file_list_path="data/splits/train_list_ALL.txt", transform=train_transform)
    val_dataset = RoadDataset(file_list_path="data/splits/val_list_ALL.txt", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. DYNAMIC MODEL & OPTIMIZER INITIALIZATION ---
    print(f"Initializing {args.model.upper()}...")
    
    if args.model == "d3s2pp":
        model = DeepLabV3PlusD3S2PP(encoder_name="resnet50", classes=1).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        is_transformer = False
    elif args.model == "mit_b3":
        model = get_model('unet', 'mit_b3').to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        is_transformer = True
    elif args.model == "resnet50":
        model = get_model('unet', 'resnet50').to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        is_transformer = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # --- 4. LOSS FUNCTIONS ---
    pos_weight = torch.tensor([5.0]).to(DEVICE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='binary')
    topo_loss_fn = TopologyAwareLoss(device=DEVICE) if args.loss == "topo" else None

    def calculate_loss(logits, targets, use_topo):
        # Shape safety
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear')
        if len(targets.shape) == 3: targets = targets.unsqueeze(1)

        probs = logits.sigmoid()
        
        # Base Pixel Loss (Switches based on architecture)
        if is_transformer:
            pixel_loss = (0.5 * bce_loss(logits, targets)) + (0.5 * dice_loss(logits, targets))
        else:
            pixel_loss = (0.75 * focal_loss(probs, targets)) + (0.25 * dice_loss(logits, targets))

        # Topology Loss Add-on
        if use_topo and topo_loss_fn:
            t_loss = topo_loss_fn(probs, targets)
            return pixel_loss + (0.1 * t_loss) # TOPO_WEIGHT = 0.1
        return pixel_loss

    def compute_iou(preds, masks):
        preds = (preds > 0.5).float()
        masks = masks.float()
        inter = (preds * masks).sum()
        union = preds.sum() + masks.sum() - inter
        return (inter + 1e-6) / (union + 1e-6)

    # --- 5. SCHEDULER & SCALER ---
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.3)
    scaler = torch.amp.GradScaler('cuda') 

    # --- 6. RESUME LOGIC ---
    start_epoch = 0
    best_iou = 0.0

    if args.resume and os.path.exists(LATEST_CHECKPOINT):
        print(f"Resuming from {LATEST_CHECKPOINT}...")
        ckpt = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
        
        if isinstance(model, nn.DataParallel): model.module.load_state_dict(ckpt['model'])
        else: model.load_state_dict(ckpt['model'])
            
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_iou = ckpt['best_iou']
        print(f"   -> Resumed at Epoch {start_epoch} (Best IoU: {best_iou:.4f})")

    # --- 7. MASTER TRAINING LOOP ---
    print(f"Starting training | Model: {args.model.upper()} | Topo Loss: {'ON' if args.loss=='topo' else 'OFF'}")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        # Warmup Logic: Only enables topo loss if past warmup epochs AND user requested it
        enable_topo = (args.loss == "topo") and (epoch >= args.warmup)
        
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = calculate_loss(logits, masks, use_topo=enable_topo)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss, val_iou_accum = 0, 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear')
                    
                    val_loss += calculate_loss(logits, masks, use_topo=(args.loss == "topo")).item()
                val_iou_accum += compute_iou(logits.sigmoid(), masks).item()
                
        avg_val_iou = val_iou_accum / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        status = f"Topo ON" if enable_topo else f"Topo OFF"
        print(f"Epoch {epoch+1}/{args.epochs} [{status}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f} | LR: {current_lr:.2e}")
        
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, avg_val_iou, current_lr])
        
        # Saves checkpionts
        save_dict = {
            'epoch': epoch,
            'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_iou': best_iou
        }
        torch.save(save_dict, LATEST_CHECKPOINT)
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            patience_counter = 0 
            torch.save(save_dict, BEST_MODEL_PATH)
            print(f"  --> Saved Best! ({best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early Stopping Triggered.")
                break

if __name__ == "__main__":
    main()