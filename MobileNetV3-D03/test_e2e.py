import os, json, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from e2e_model import EndToEndVQAModel
from vqa_dataset_e2e import get_dataloaders_e2e

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    print("Loading test dataset...")
    _, _, test_dl = get_dataloaders_e2e(splits_dir, videos_dir, batch_size=8)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model  = EndToEndVQAModel().to(device)
    best_path = os.path.join(WORK_DIR, "best_model_e2e.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No checkpoint found at {best_path}. Run train_e2e.py first.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Loaded: {best_path}")

    keys = ['mos','quality_class','spatial_flag','hallucination_flag','lighting_flag',
            'rendering_flag','physics_violation_flag','object_flicker_flag','motion_inconsistency_flag']
    all_p = {k:[] for k in keys}
    all_t = {k:[] for k in keys}

    print(f"Inference on {len(test_dl)} test batches...")
    with torch.no_grad():
        for videos, targets in test_dl:
            videos  = videos.to(device)
            outputs = model(videos)
            for k in keys:
                all_p[k].append(outputs[k].cpu())
                all_t[k].append(targets[k].cpu())

    for k in keys:
        all_p[k] = torch.cat(all_p[k], 0).numpy().flatten()
        all_t[k] = torch.cat(all_t[k], 0).numpy().flatten()

    mos_p    = all_p['mos'] * 100.0
    mos_t    = all_t['mos'] * 100.0
    mos_rmse = np.sqrt(mean_squared_error(mos_t, mos_p))
    mos_plcc = pearsonr(mos_t,  mos_p)[0] if len(np.unique(mos_p)) > 1 else 0.0
    mos_srcc = spearmanr(mos_t, mos_p)[0] if len(np.unique(mos_p)) > 1 else 0.0
    qc_p     = np.clip(np.round(all_p['quality_class'] * 5.0), 1, 5)
    qc_t     = np.round(all_t['quality_class'] * 5.0)
    qc_acc   = accuracy_score(qc_t, qc_p)

    flag_keys = ['spatial_flag','hallucination_flag','lighting_flag','rendering_flag',
                 'physics_violation_flag','object_flicker_flag','motion_inconsistency_flag']
    flag_f1s = {}
    for k in flag_keys:
        probs = 1.0 / (1.0 + np.exp(-all_p[k]))
        flag_f1s[k] = f1_score(all_t[k].astype(int), (probs > 0.5).astype(int), zero_division=0)

    macro = np.mean(list(flag_f1s.values()))

    results = {
        "Config": {"dropout": 0.3, "patience": 20},
        "MOS Metrics": {"mos_SRCC": float(mos_srcc), "mos_PLCC": float(mos_plcc), "mos_RMSE": float(mos_rmse)},
        "Quality Classification": {"qualityclass_accuracy": float(qc_acc)},
        "Per-flag F1 scores": {
            "spatial_flag_f1":              float(flag_f1s['spatial_flag']),
            "hallucination_flag_f1":        float(flag_f1s['hallucination_flag']),
            "lighting_flag_f1":             float(flag_f1s['lighting_flag']),
            "rendering_flag_f1":            float(flag_f1s['rendering_flag']),
            "physical_violation_flag_f1":   float(flag_f1s['physics_violation_flag']),
            "motion_inconsistency_flag_f1": float(flag_f1s['motion_inconsistency_flag']),
            "object_flicker_flag_f1":       float(flag_f1s['object_flicker_flag'])
        },
        "Overall": {"macro_avg_flag_f1": float(macro)}
    }

    out_json = os.path.join(WORK_DIR, "test_results.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*55)
    print(f"FINAL TEST METRICS  [Dropout=0.3 | Patience=20]".center(55))
    print("="*55)
    print(f"  MOS SRCC : {mos_srcc:.4f}")
    print(f"  MOS PLCC : {mos_plcc:.4f}")
    print(f"  MOS RMSE : {mos_rmse:.4f}")
    print(f"  QC Acc   : {qc_acc:.4f}")
    print("\nPer-flag F1 Scores:")
    for k,v in results['Per-flag F1 scores'].items():
        print(f"  {k:<35}: {v:.4f}")
    print(f"\n  Macro Avg Flag F1 : {macro:.4f}")
    print("="*55)

    out_plot = os.path.join(WORK_DIR, "mos_scatter_plot.png")
    plt.figure(figsize=(8,6))
    plt.scatter(mos_t, mos_p, alpha=0.4, color='royalblue', s=10)
    lo, hi = min(mos_t.min(), mos_p.min()), max(mos_t.max(), mos_p.max())
    plt.plot([lo,hi],[lo,hi],'r--',lw=2)
    plt.title("Actual vs Predicted MOS (Test Set) — Dropout=0.3", fontsize=13)
    plt.xlabel("Actual MOS"); plt.ylabel("Predicted MOS")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved: {out_json}\nSaved: {out_plot}")

if __name__ == "__main__":
    main()
