#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from models.eg_dino import DINOv2EdgeAware
from utils.edge_dataset import EdgeAwareDataset, precompute_hed_edges
from utils.metrics import calculate_cldice, calculate_boundary_f1

def set_seed(seed=42):
    """Ensures absolute reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_validation(model, loader, device, num_classes, img_height, img_width):
    """Validation loop using standard Fixed (Argmax) metrics."""
    model.eval()
    conf_matrix = torch.zeros(num_classes, num_classes).to(device)
    acc_cldice = []
    acc_bf1_strict = []
    acc_bf1_tol2 = []
    
    with torch.no_grad():
        for imgs, masks, edges in tqdm(loader, desc="Validation", leave=False):
            imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
            outputs = model(imgs, edges)
            preds = outputs.argmax(dim=1)
            
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)
            indices = masks_flat * num_classes + preds_flat
            bincount = torch.bincount(indices, minlength=num_classes**2)
            conf_matrix += bincount.view(num_classes, num_classes).float()

            pred_np = preds.cpu().numpy()
            gt_np = masks.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p_i, g_i = pred_np[i], gt_np[i]
                acc_cldice.append(calculate_cldice(p_i, g_i))
                acc_bf1_strict.append(calculate_boundary_f1(p_i, g_i, tolerance=0))
                acc_bf1_tol2.append(calculate_boundary_f1(p_i, g_i, tolerance=2))

    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    iou = tp / (tp + fp + fn + 1e-6)
    crack_iou = iou[1].item()
    crack_dice = (2. * tp[1] / (2. * tp[1] + fp[1] + fn[1] + 1e-6)).item()
    
    print("\n" + "="*65)
    print(f"      VALIDATION REPORT ({img_height}x{img_width})")
    print("="*65)
    print(f"{'Metric':<25} | {'Score':<10}")
    print("-" * 65)
    print(f"{'IoU (Crack, Global)':<25} | {crack_iou:.4f}")
    print(f"{'Dice (Crack, Global)':<25} | {crack_dice:.4f}")
    print("-" * 65)
    print(f"{'Boundary F1 (Tol=2)':<25} | {np.mean(acc_bf1_tol2):.4f}")
    print(f"{'Boundary F1 (Strict)':<25} | {np.mean(acc_bf1_strict):.4f}")
    print("-" * 65)
    print(f"{'clDice (Topology)':<25} | {np.mean(acc_cldice):.4f}")
    print("="*65)
    
    return crack_iou

def main():
    parser = argparse.ArgumentParser(description="Train EG-DINO for Edge-Guided Segmentation")
    parser.add_argument('--data_root', type=str, default='./dataset_crack500', help='Path to dataset root')
    parser.add_argument('--img_height', type=int, default=630, help='Target image height')
    parser.add_argument('--img_width', type=int, default=350, help='Target image width')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr_encoder', type=float, default=5e-6, help='Learning rate for DINOv2 backbone')
    parser.add_argument('--lr_decoder', type=float, default=5e-4, help='Learning rate for decoder')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-base', help='HuggingFace DINOv2 model name')
    parser.add_argument('--save_dir', type=str, default='./weights', help='Directory to save model weights')
    parser.add_argument('--recompute_edges', action='store_true', help='Force recomputation of HED edges')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, f"eg_dino_{args.img_height}x{args.img_width}.pth")

    print(f"--- 1. Checking HED Edge Cache ({args.img_height}x{args.img_width}) ---")
    train_edge_dir = precompute_hed_edges(args.data_root, "train", args.img_height, args.img_width, args.recompute_edges, device)
    val_edge_dir = precompute_hed_edges(args.data_root, "val", args.img_height, args.img_width, args.recompute_edges, device)

    train_dataset = EdgeAwareDataset(args.data_root, "train", args.img_height, args.img_width, edge_dir=train_edge_dir)
    val_dataset = EdgeAwareDataset(args.data_root, "val", args.img_height, args.img_width, edge_dir=val_edge_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    print("--- 2. Initializing EG-DINO Model ---")
    num_classes = 2
    model = DINOv2EdgeAware(args.dino_model, num_classes, args.img_height, args.img_width).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr_encoder}, 
        {'params': model.decoder.parameters(), 'lr': args.lr_decoder}
    ])
    
    # Using moderate weights tensor as defined in your original script
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], dtype=torch.float32).to(device))
    best_iou = -1.0

    print("--- 3. Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for imgs, masks, edges in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, edges)
            loss = ce_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        val_iou = evaluate_validation(model, val_loader, device, num_classes, args.img_height, args.img_width)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved Best Model (IoU: {best_iou:.4f}) to {model_save_path}")

if __name__ == '__main__':
    main()