import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error

from e2e_model import EndToEndVQAModel, compute_loss
from vqa_dataset_e2e import get_dataloaders_e2e

# ── Working directory is always the script's own folder ─────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_metrics(preds_dict, targets_dict):
    metrics = {}

    mos_p = preds_dict['mos'].numpy().flatten() * 100.0
    mos_t = targets_dict['mos'].numpy().flatten() * 100.0
    metrics['mos_rmse'] = np.sqrt(mean_squared_error(mos_t, mos_p))

    if len(np.unique(mos_p)) > 1 and len(np.unique(mos_t)) > 1:
        metrics['mos_plcc'] = pearsonr(mos_t, mos_p)[0]
        metrics['mos_srcc'] = spearmanr(mos_t, mos_p)[0]
    else:
        metrics['mos_plcc'] = 0.0
        metrics['mos_srcc'] = 0.0

    qc_p = preds_dict['quality_class'].numpy().flatten() * 5.0
    qc_t = targets_dict['quality_class'].numpy().flatten() * 5.0
    metrics['quality_class_rmse'] = np.sqrt(mean_squared_error(qc_t, qc_p))

    flag_keys = [
        'spatial_flag', 'hallucination_flag', 'lighting_flag',
        'rendering_flag', 'physics_violation_flag',
        'object_flicker_flag', 'motion_inconsistency_flag'
    ]

    for key in flag_keys:
        preds       = preds_dict[key]
        probs       = torch.sigmoid(preds).numpy().flatten()
        binary_preds   = (probs > 0.5).astype(int)
        binary_targets = targets_dict[key].numpy().flatten().astype(int)
        metrics[f"{key}_f1"] = f1_score(binary_targets, binary_preds, zero_division=0)

    return metrics


def run_epoch(model, dataloader, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()

    epoch_losses = []
    all_preds   = {k: [] for k in ['mos', 'quality_class', 'spatial_flag', 'hallucination_flag',
                                    'lighting_flag', 'rendering_flag', 'physics_violation_flag',
                                    'object_flicker_flag', 'motion_inconsistency_flag']}
    all_targets = {k: [] for k in all_preds.keys()}

    with torch.set_grad_enabled(is_train):
        for videos, targets in dataloader:
            videos  = videos.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            if is_train:
                optimizer.zero_grad()

            predictions      = model(videos)
            loss, _          = compute_loss(predictions, targets)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_losses.append(loss.item())

            for k in all_preds.keys():
                all_preds[k].append(predictions[k].detach().cpu())
                all_targets[k].append(targets[k].detach().cpu())

    for k in all_preds.keys():
        all_preds[k]   = torch.cat(all_preds[k],   dim=0)
        all_targets[k] = torch.cat(all_targets[k], dim=0)

    avg_loss            = np.mean(epoch_losses)
    metrics             = calculate_metrics(all_preds, all_targets)
    metrics['total_loss'] = avg_loss

    return metrics


def main():
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"

    print("Loading datasets...")
    train_dl, val_dl, _ = get_dataloaders_e2e(splits_dir, videos_dir, batch_size=6)
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model  = EndToEndVQAModel().to(device)

    # Print param summary
    total_params    = sum(p.numel() for p in model.parameters())
    frozen_params   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params= total_params - frozen_params
    print(f"\nBackbone output dim : 960 → projected to 1280")
    print(f"Total parameters    : {total_params:,}")
    print(f"Frozen parameters   : {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Differential Learning Rates
    backbone_params, mlp_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone_features' in name or 'proj_960_to_1280' in name:
            backbone_params.append(param)
        else:
            mlp_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': mlp_params,      'lr': 3e-4}
    ], weight_decay=1e-3)

    scheduler      = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    num_epochs     = 50
    best_srcc      = -float('inf')
    patience       = 10
    no_improve     = 0
    history        = []
    start_epoch    = 1

    checkpoint_path  = os.path.join(WORK_DIR, "checkpoint_latest.pth")
    best_model_path  = os.path.join(WORK_DIR, "best_model_e2e.pth")
    log_path         = os.path.join(WORK_DIR, "training_log_e2e.csv")

    # ── Auto-Resume ──────────────────────────────────────────────
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_srcc   = ckpt['best_val_mos_srcc']
        no_improve  = ckpt.get('epochs_no_improve', 0)
        if os.path.exists(log_path):
            history = pd.read_csv(log_path).to_dict('records')

    elif os.path.exists(best_model_path):
        print(f"Loading weights from {best_model_path} ...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        if os.path.exists(log_path):
            history     = pd.read_csv(log_path).to_dict('records')
            start_epoch = int(history[-1]['epoch']) + 1

    print("\nStarting Training...\n")

    for epoch in range(start_epoch, num_epochs + 1):
        lr_bb  = optimizer.param_groups[0]['lr']
        lr_mlp = optimizer.param_groups[1]['lr']
        print(f"\n[Epoch {epoch}/{num_epochs}] LR Backbone: {lr_bb:.2e} | LR MLP: {lr_mlp:.2e}")

        train_m = run_epoch(model, train_dl, optimizer, device, is_train=True)
        val_m   = run_epoch(model, val_dl,   optimizer, device, is_train=False)

        scheduler.step()

        # Print table
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'-'*70}")
        print(f"{'Metric':<30} | {'Train':<15} | {'Validation':<15}")
        print(f"{'-'*70}")
        keys_to_print = ['total_loss', 'mos_rmse', 'mos_plcc', 'mos_srcc', 'quality_class_rmse'] + \
                        [k for k in train_m.keys() if '_f1' in k]
        for k in keys_to_print:
            print(f"{k:<30} | {train_m[k]:<15.4f} | {val_m[k]:<15.4f}")
        print(f"{'='*70}")

        val_srcc = val_m['mos_srcc']
        if val_srcc > best_srcc:
            print(f"*** New Best Val SRCC: {val_srcc:.4f} (prev: {best_srcc:.4f}) — saving ***")
            best_srcc  = val_srcc
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s). Best SRCC: {best_srcc:.4f}")

        # Save continuous checkpoint
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_mos_srcc':    best_srcc,
            'epochs_no_improve':    no_improve
        }, checkpoint_path)

        # Log to CSV
        row = {'epoch': epoch}
        for k, v in train_m.items():
            row[f'train_{k}'] = v
        for k, v in val_m.items():
            row[f'val_{k}'] = v
        history.append(row)
        pd.DataFrame(history).to_csv(log_path, index=False)

        if no_improve >= patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch}.")
            break

    print("\nTraining Complete!")
    print(f"Best Validation MOS SRCC: {best_srcc:.4f}")
    print(f"Full log saved to: {log_path}")


if __name__ == "__main__":
    main()
