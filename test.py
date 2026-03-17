#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.eg_dino import DINOv2EdgeAware
from utils.edge_dataset import EdgeAwareDataset, precompute_hed_edges
from utils.metrics import ODSCalculator, get_cldice_components

def main():
    parser = argparse.ArgumentParser(description="Test EG-DINO and compute ODS & Fixed Metrics")
    parser.add_argument('--data_root', type=str, default='./dataset_crack500', help='Path to dataset root')
    parser.add_argument('--test_split', type=str, default='test', help='Split to test on (e.g., test, val)')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained .pth model file')
    parser.add_argument('--img_height', type=int, default=630, help='Target image height')
    parser.add_argument('--img_width', type=int, default=350, help='Target image width')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-base', help='HuggingFace DINOv2 model name')
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"❌ Error: Model file '{args.weights}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2

    print(f"--- Setting up Test on '{args.test_split}' Split ({args.img_height}x{args.img_width}) ---")
    edge_dir = precompute_hed_edges(args.data_root, args.test_split, args.img_height, args.img_width, device=device)
    
    # We use batch_size=1 for precise test calculation per your original script
    dataset = EdgeAwareDataset(args.data_root, args.test_split, args.img_height, args.img_width, edge_dir=edge_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("⏳ Loading DINOv2 + HED Model...")
    model = DINOv2EdgeAware(args.dino_model, num_classes, args.img_height, args.img_width).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    
    conf_matrix_fixed = torch.zeros(num_classes, num_classes).to(device)
    ods = ODSCalculator(device=device)
    g_op_f, g_tp_f, g_os_f, g_ts_f = 0, 0, 0, 0 
    
    print("🚀 Starting Inference...")
    with torch.no_grad():
        for imgs, masks, edges, names in tqdm(loader, desc="Testing"):
            imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
            outputs = model(imgs, edges)
            
            # 1. Softmax for ODS Tracker
            probs = torch.softmax(outputs, dim=1)
            crack_probs = probs[:, 1, :, :] 
            ods.update(crack_probs, masks) 

            # 2. Fixed Argmax Tracking
            preds_fixed = outputs.argmax(dim=1) 
            indices = masks.view(-1) * num_classes + preds_fixed.view(-1)
            conf_matrix_fixed += torch.bincount(indices, minlength=num_classes**2).view(num_classes, num_classes).float()
            
            # 3. Fixed clDice Accumulation
            pred_np = preds_fixed.cpu().numpy()[0]
            gt_np = masks.cpu().numpy()[0]
            o_p, t_p, o_s, t_s = get_cldice_components(pred_np, gt_np)
            g_op_f += o_p; g_tp_f += t_p; g_os_f += o_s; g_ts_f += t_s

    # --- FINAL METRIC CALCULATION ---
    tp = conf_matrix_fixed.diag()
    fp = conf_matrix_fixed.sum(dim=0) - tp
    fn = conf_matrix_fixed.sum(dim=1) - tp
    eps = 1e-6
    
    fixed_p = (tp[1] / (tp[1] + fp[1] + eps)).item()
    fixed_r = (tp[1] / (tp[1] + fn[1] + eps)).item()
    fixed_f1 = 2 * fixed_p * fixed_r / (fixed_p + fixed_r + eps)
    fixed_iou_c = (tp[1] / (tp[1] + fp[1] + fn[1] + eps)).item()
    fixed_miou = ((fixed_iou_c + (tp[0] / (tp[0] + fp[0] + fn[0] + eps)).item()) / 2.0)
    
    cl_prec_f = g_op_f / (g_tp_f + eps)
    cl_sens_f = g_os_f / (g_ts_f + eps)
    fixed_cldice = 2 * (cl_prec_f * cl_sens_f) / (cl_prec_f + cl_sens_f + eps)

    # Get ODS specific results
    ods_res = ods.get_best_metrics()
    ods_cldice = ods.compute_cldice_at_threshold(ods_res['Threshold'])

    results_data = {
        "Metric": ["Precision", "Recall", "F1-Score", "Crack IoU", "Mean IoU", "clDice"],
        "Fixed (Argmax)": [
            f"{fixed_p:.4f}", f"{fixed_r:.4f}", f"{fixed_f1:.4f}", f"{fixed_iou_c:.4f}", f"{fixed_miou:.4f}", f"{fixed_cldice:.4f}"
        ],
        "ODS (Best Thresh)": [
            f"{ods_res['Precision']:.4f}", f"{ods_res['Recall']:.4f}", f"{ods_res['F1-Score']:.4f}", 
            f"{ods_res['Crack IoU']:.4f}", f"{ods_res['Mean IoU']:.4f}", f"{ods_cldice:.4f}"
        ]
    }
    
    df = pd.DataFrame(results_data)
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS: {args.data_root} ({args.test_split})")
    print("="*60)
    print(df.to_string(index=False))
    print("-" * 60)
    print(f"Optimal Threshold (ODS): {ods_res['Threshold']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()