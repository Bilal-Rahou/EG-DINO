import torch
import numpy as np
import cv2
from skimage.morphology import skeletonize
from tqdm import tqdm

def cl_score(v, s):
    skeleton = skeletonize(v)
    return np.sum(skeleton * s) / (np.sum(skeleton) + 1e-8) if skeleton.sum() > 0 else 0.0

def calculate_cldice(v_p, v_l):
    if v_p.sum() == 0 and v_l.sum() == 0: return 1.0
    if v_p.sum() == 0 or v_l.sum() == 0: return 0.0
    tprec = cl_score(v_p, v_l)
    tsens = cl_score(v_l, v_p)
    return 2 * tprec * tsens / (tprec + tsens + 1e-8)

def get_cldice_components(v_p, v_l):
    """Returns raw skeleton intersections and totals for global calculation."""
    if np.sum(v_p) == 0 and np.sum(v_l) == 0: 
        return 0, 0, 0, 0 
        
    skeleton = skeletonize(v_p > 0)
    skeleton_gt = skeletonize(v_l > 0)
    
    overlap_prec = np.sum(skeleton & (v_l > 0))
    total_prec = np.sum(skeleton)
    
    overlap_sens = np.sum(skeleton_gt & (v_p > 0))
    total_sens = np.sum(skeleton_gt)
    
    return overlap_prec, total_prec, overlap_sens, total_sens

def calculate_boundary_f1(pred, gt, tolerance=2):
    def get_boundary(mask):
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(mask)
        cv2.drawContours(boundary, contours, -1, 1, 1)
        return boundary
        
    pred_b, gt_b = get_boundary(pred), get_boundary(gt)
    if pred_b.sum() == 0 and gt_b.sum() == 0: return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0: return 0.0
    
    if tolerance > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
        gt_d, pred_d = cv2.dilate(gt_b, kernel), cv2.dilate(pred_b, kernel)
        prec = np.sum(pred_b * gt_d) / (np.sum(pred_b) + 1e-8)
        rec = np.sum(gt_b * pred_d) / (np.sum(gt_b) + 1e-8)
    else:
        inter = np.sum(pred_b * gt_b)
        prec, rec = inter / (np.sum(pred_b)+1e-8), inter / (np.sum(gt_b)+1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)

class ODSCalculator:
    def __init__(self, num_thresholds=255, device='cuda'):
        self.thresholds = torch.linspace(0, 1, steps=num_thresholds).to(device)
        self.TP = torch.zeros(num_thresholds).to(device)
        self.FP = torch.zeros(num_thresholds).to(device)
        self.FN = torch.zeros(num_thresholds).to(device)
        self.TN = torch.zeros(num_thresholds).to(device)
        self.device = device
        self.stored_probs = []
        self.stored_gts = []

    def update(self, preds_prob, targets):
        self.stored_probs.append(preds_prob.cpu())
        self.stored_gts.append(targets.cpu())
        
        preds_flat = preds_prob.view(-1)
        targets_flat = targets.view(-1)
        binary_preds = preds_flat.unsqueeze(0) >= self.thresholds.unsqueeze(1)
        gt = targets_flat.unsqueeze(0).bool()

        self.TP += (binary_preds & gt).sum(dim=1)
        self.FP += (binary_preds & ~gt).sum(dim=1)
        self.FN += (~binary_preds & gt).sum(dim=1)
        self.TN += (~binary_preds & ~gt).sum(dim=1)

    def get_best_metrics(self):
        eps = 1e-6
        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        iou_crack = self.TP / (self.TP + self.FP + self.FN + eps)
        iou_bg = self.TN / (self.TN + self.FP + self.FN + eps)
        mean_iou = (iou_crack + iou_bg) / 2.0
        best_idx = torch.argmax(f1)
        return {
            "Threshold": self.thresholds[best_idx].item(),
            "Precision": precision[best_idx].item(),
            "Recall": recall[best_idx].item(),
            "F1-Score": f1[best_idx].item(),
            "Crack IoU": iou_crack[best_idx].item(),
            "Mean IoU": mean_iou[best_idx].item()
        }
        
    def compute_cldice_at_threshold(self, best_thresh):
        print(f"⏳ Computing clDice at Optimal Threshold: {best_thresh:.4f} ...")
        g_op, g_tp, g_os, g_ts = 0, 0, 0, 0 
        
        for prob_batch, gt_batch in tqdm(zip(self.stored_probs, self.stored_gts), total=len(self.stored_probs), desc="ODS clDice", leave=False):
            pred_mask = (prob_batch > best_thresh).int() 
            for i in range(pred_mask.shape[0]):
                p_np = pred_mask[i].numpy().squeeze().astype(np.uint8)
                g_np = gt_batch[i].numpy().squeeze().astype(np.uint8)
                
                o_p, t_p, o_s, t_s = get_cldice_components(p_np, g_np)
                g_op += o_p; g_tp += t_p; g_os += o_s; g_ts += t_s
                
        eps = 1e-6
        cl_prec = g_op / (g_tp + eps)
        cl_sens = g_os / (g_ts + eps)
        return 2 * (cl_prec * cl_sens) / (cl_prec + cl_sens + eps)