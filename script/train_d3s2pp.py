import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import segmentation_models_pytorch as smp
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from initial.dataset import RoadDataset


# --- CONFIGURATION ---
DATA_DIR = '../src/AOI_2_Vegas'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
BATCH_SIZE = 4
PATIENCE = 10
CHECKPOINT_FILE = "checkpoint_d3s2pp.pth"
MODEL_SAVE_PATH = "d3s2pp_resnet50_best.pth"

# --- 1. DEFINE CUSTOM D3S2PP MODULE ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, 
                                   dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class D3S2PP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.ac1 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[0], padding=atrous_rates[0])
        self.ac2 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[1], padding=atrous_rates[1])
        self.ac3 = DepthwiseSeparableConv(in_channels, out_channels, dilation=atrous_rates[2], padding=atrous_rates[2])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        res.append(self.conv1x1(x))
        res.append(self.ac1(x))
        res.append(self.ac2(x))
        res.append(self.ac3(x))
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3PlusD3S2PP(SegmentationModel):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", classes=1):
        super().__init__()
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        
        encoder_channels = self.encoder.out_channels
        self.d3s2pp = D3S2PP(in_channels=encoder_channels[-1], out_channels=256)
        
        # FIX: Pass all required arguments for DeepLabV3PlusDecoder
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            encoder_depth=5,          
            aspp_separable=True,      
            aspp_dropout=0.5,         
        )
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, classes, kernel_size=1),
        )
        self.classification_head = None

    def forward(self, x):
        features = self.encoder(x)
        
        # 1. Custom D3S2PP Processing
        high_level_features = features[-1]
        aspp_output = self.d3s2pp(high_level_features)
        
        # 2. Manual Decoder Logic 
        # Upsample D3S2PP output by 4
        x_high = F.interpolate(aspp_output, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Low Level Features (from encoder stage 2, index 1)
        low_level_features = features[1] 
        
        # Project Low Level
        if not hasattr(self, 'low_level_project'):
            self.low_level_project = nn.Sequential(
                nn.Conv2d(low_level_features.shape[1], 48, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU()
            ).to(x.device)
            
        x_low = self.low_level_project(low_level_features)
        
        # Concat
        if x_high.shape[-2:] != x_low.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=False)
            
        x_combined = torch.cat([x_high, x_low], dim=1)
        
        # Final Convs
        if not hasattr(self, 'final_convs'):
            self.final_convs = nn.Sequential(
                nn.Conv2d(304, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ).to(x.device)
            
        x_out = self.final_convs(x_combined)
        
        # Upsample to original size
        logits = self.segmentation_head(x_out)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        
        return logits

# --- DATA PREP ---
train_transform = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=0), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
])

print("Loading dataset...")

train_dataset = RoadDataset(file_list_path="../src/train_list.txt", transform=train_transform)
val_dataset = RoadDataset(file_list_path="../src/val_list.txt", transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- MODEL SETUP ---
print("Initializing Custom D3S2PP Model...")
model = DeepLabV3PlusD3S2PP(encoder_name="resnet50", classes=1).to(DEVICE)

# --- OPTIMIZER & LOSS ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
focal_loss = smp.losses.FocalLoss(mode='binary')

def loss_fn(logits, targets):
    # Ensure targets match logits shape
    if logits.shape != targets.shape:
        # Check if we just need to align spatial dimensions
        if logits.shape[-2:] != targets.shape[-2:]:
             logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        # Check if targets missing channel dim
        if len(targets.shape) == 3 and len(logits.shape) == 4:
             targets = targets.unsqueeze(1)
             
    d_loss = dice_loss(logits, targets)
    probs = logits.sigmoid()
    f_loss = focal_loss(probs, targets)
    return (0.75 * f_loss) + (0.25 * d_loss)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-4, 
    total_steps=EPOCHS * len(train_loader),
    pct_start=0.3
)

scaler = torch.amp.GradScaler('cuda') 

def get_iou_stats(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    masks = masks.float()
    intersection = (preds * masks).sum() 
    union = preds.sum() + masks.sum() - intersection
    return intersection.item(), union.item()

# --- TRAINING LOOP ---
print(f"Starting training... (Patience: {PATIENCE})")

best_iou = 0.0
patience_counter = 0
start_epoch = 0

if os.path.exists(CHECKPOINT_FILE):
    print("🔄 Resuming from checkpoint...")
    ckpt = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
    best_iou = ckpt['best_iou']

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0
    
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = loss_fn(logits, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        epoch_loss += loss.item()
        
    avg_train_loss = epoch_loss / len(train_loader)
    
    model.eval()
    total_intersection = 0.0
    total_union = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(images)
                # FIX: Resize logits to match masks shape before sigmoid
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                preds = logits.sigmoid()
            
            i, u = get_iou_stats(preds, masks)
            total_intersection += i
            total_union += u
            
    if total_union == 0:
        avg_val_iou = 1.0
    else:
        avg_val_iou = total_intersection / total_union

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")
    
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_iou': best_iou
    }, CHECKPOINT_FILE)
    
    if (avg_val_iou > best_iou) and (total_intersection > 100):
        best_iou = avg_val_iou
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> New Best Model Saved! ({best_iou:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("🛑 Early Stopping.")
            break

print("Training finished.")