import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score

from e2e_model import EndToEndVQAModel
from vqa_dataset_e2e import get_dataloaders_e2e

def main():
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    
    print("Loading test dataset...")
    # Get test dataloader
    _, _, test_dl = get_dataloaders_e2e(splits_dir, videos_dir, batch_size=8)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize and load model
    model = EndToEndVQAModel().to(device)
    model_path = "best_model_e2e.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}")
        
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = {k: [] for k in ['mos', 'quality_class', 'spatial_flag', 'hallucination_flag', 
                                 'lighting_flag', 'rendering_flag', 'physics_violation_flag', 
                                 'object_flicker_flag', 'motion_inconsistency_flag']}
    all_targets = {k: [] for k in all_preds.keys()}
    
    print(f"Running inference on {len(test_dl)} test batches...")
    
    with torch.no_grad():
        for videos, targets in test_dl:
            videos = videos.to(device)
            outputs = model(videos)
            
            for k in all_preds.keys():
                all_preds[k].append(outputs[k].cpu())
                all_targets[k].append(targets[k].cpu())
                
    # Concatenate all batches
    for k in all_preds.keys():
        all_preds[k] = torch.cat(all_preds[k], dim=0).numpy().flatten()
        all_targets[k] = torch.cat(all_targets[k], dim=0).numpy().flatten()
        
    # Calculate MOS Metrics (Denormalize by multiplying by 100)
    mos_p = all_preds['mos'] * 100.0
    mos_t = all_targets['mos'] * 100.0
    
    mos_rmse = np.sqrt(mean_squared_error(mos_t, mos_p))
    mos_plcc = pearsonr(mos_t, mos_p)[0] if len(np.unique(mos_p)) > 1 else 0.0
    mos_srcc = spearmanr(mos_t, mos_p)[0] if len(np.unique(mos_p)) > 1 else 0.0
    
    # Calculate Quality Class Accuracy (Denormalize by multiplying by 5)
    qc_p = np.clip(np.round(all_preds['quality_class'] * 5.0), 1, 5)
    qc_t = np.round(all_targets['quality_class'] * 5.0)
    qc_acc = accuracy_score(qc_t, qc_p)
    
    # Calculate Flag Metrics
    flag_keys = [
        'spatial_flag', 'hallucination_flag', 'lighting_flag',
        'rendering_flag', 'physics_violation_flag', 
        'object_flicker_flag', 'motion_inconsistency_flag'
    ]
    
    flag_f1_scores = {}
    for key in flag_keys:
        # Apply sigmoid and threshold at 0.5
        probs = 1.0 / (1.0 + np.exp(-all_preds[key])) # sigmoid
        binary_preds = (probs > 0.5).astype(int)
        binary_targets = all_targets[key].astype(int)
        
        f1 = f1_score(binary_targets, binary_preds, zero_division=0)
        flag_f1_scores[key] = f1
        
    macro_avg_flag_f1 = np.mean(list(flag_f1_scores.values()))
    
    # Structure Results
    results = {
        "MOS Metrics": {
            "mos_SRCC": float(mos_srcc),
            "mos_PLCC": float(mos_plcc),
            "mos_RMSE": float(mos_rmse)
        },
        "Quality Classification": {
            "qualityclass_accuracy": float(qc_acc)
        },
        "Per-flag F1 scores": {
            "spatial_flag_f1": float(flag_f1_scores['spatial_flag']),
            "hallucination_flag_f1": float(flag_f1_scores['hallucination_flag']),
            "lighting_flag_f1": float(flag_f1_scores['lighting_flag']),
            "rendering_flag_f1": float(flag_f1_scores['rendering_flag']),
            "physical_violation_flag_f1": float(flag_f1_scores['physics_violation_flag']),
            "motion_inconsistency_flag_f1": float(flag_f1_scores['motion_inconsistency_flag']),
            "object_flicker_flag_f1": float(flag_f1_scores['object_flicker_flag'])
        },
        "Overall": {
            "macro_avg_flag_f1": float(macro_avg_flag_f1)
        }
    }
    
    # Save to JSON
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Print Table
    print("\n" + "="*50)
    print("FINAL TEST METRICS".center(50))
    print("="*50)
    print("MOS Metrics:")
    print(f"  - SRCC: {mos_srcc:.4f}")
    print(f"  - PLCC: {mos_plcc:.4f}")
    print(f"  - RMSE: {mos_rmse:.4f}")
    print("\nQuality Classification:")
    print(f"  - Accuracy: {qc_acc:.4f}")
    print("\nPer-flag F1 Scores:")
    for k, v in results['Per-flag F1 scores'].items():
        print(f"  - {k:<30}: {v:.4f}")
    print("\nOverall:")
    print(f"  - Macro Avg Flag F1: {macro_avg_flag_f1:.4f}")
    print("="*50 + "\n")
    
    # Generate Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_t, mos_p, alpha=0.5, color='royalblue')
    
    # Plot perfect prediction line y=x
    min_val = min(np.min(mos_t), np.min(mos_p))
    max_val = max(np.max(mos_t), np.max(mos_p))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title("Actual vs Predicted MOS Scores (Test Set)", fontsize=14)
    plt.xlabel("Actual MOS", fontsize=12)
    plt.ylabel("Predicted MOS", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("mos_scatter_plot.png", dpi=300)
    print("Saved scatter plot to 'mos_scatter_plot.png'")
    print("Saved JSON results to 'test_results.json'")

if __name__ == "__main__":
    main()
