import os, torch, torch.nn as nn, torch.optim as optim, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error
from e2e_model import EndToEndVQAModel, compute_loss
from vqa_dataset_e2e import get_dataloaders_e2e

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_metrics(preds, tgts):
    m = {}
    p = preds['mos'].numpy().flatten() * 100.0
    t = tgts['mos'].numpy().flatten()  * 100.0
    m['mos_rmse'] = np.sqrt(mean_squared_error(t, p))
    m['mos_plcc'] = pearsonr(t, p)[0]  if len(np.unique(p)) > 1 else 0.0
    m['mos_srcc'] = spearmanr(t, p)[0] if len(np.unique(p)) > 1 else 0.0
    qp = preds['quality_class'].numpy().flatten() * 5.0
    qt = tgts['quality_class'].numpy().flatten()  * 5.0
    m['quality_class_rmse'] = np.sqrt(mean_squared_error(qt, qp))
    for key in ['spatial_flag','hallucination_flag','lighting_flag','rendering_flag',
                'physics_violation_flag','object_flicker_flag','motion_inconsistency_flag']:
        probs = torch.sigmoid(preds[key]).numpy().flatten()
        bp = (probs > 0.5).astype(int)
        bt = tgts[key].numpy().flatten().astype(int)
        m[f'{key}_f1'] = f1_score(bt, bp, zero_division=0)
    return m

def run_epoch(model, dl, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    losses, all_p, all_t = [], \
        {k:[] for k in ['mos','quality_class','spatial_flag','hallucination_flag',
                         'lighting_flag','rendering_flag','physics_violation_flag',
                         'object_flicker_flag','motion_inconsistency_flag']}, \
        {k:[] for k in ['mos','quality_class','spatial_flag','hallucination_flag',
                         'lighting_flag','rendering_flag','physics_violation_flag',
                         'object_flicker_flag','motion_inconsistency_flag']}
    with torch.set_grad_enabled(is_train):
        for videos, targets in dl:
            videos  = videos.to(device)
            targets = {k: v.to(device) for k,v in targets.items()}
            if is_train: optimizer.zero_grad()
            preds = model(videos)
            loss, _ = compute_loss(preds, targets)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
            for k in all_p:
                all_p[k].append(preds[k].detach().cpu())
                all_t[k].append(targets[k].detach().cpu())
    for k in all_p:
        all_p[k] = torch.cat(all_p[k], 0)
        all_t[k] = torch.cat(all_t[k], 0)
    m = calculate_metrics(all_p, all_t)
    m['total_loss'] = np.mean(losses)
    return m

def main():
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    print("Loading datasets..."); train_dl, val_dl, _ = get_dataloaders_e2e(splits_dir, videos_dir, batch_size=6)
    print(f"Train: {len(train_dl)} batches | Val: {len(val_dl)} batches")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model  = EndToEndVQAModel().to(device)

    total     = sum(p.numel() for p in model.parameters())
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = total - frozen
    print(f"Config: Dropout=0.3 | Patience=20")
    print(f"Total={total:,}  Frozen={frozen:,}  Trainable={trainable:,}\n")

    bb_params, mlp_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'backbone' in name:
            bb_params.append(param)
        else:
            mlp_params.append(param)

    optimizer = optim.AdamW([{'params': bb_params, 'lr': 1e-5},
                              {'params': mlp_params,'lr': 3e-4}], weight_decay=1e-3)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_srcc, patience, no_improve, history, start_epoch = -float('inf'), 20, 0, [], 1
    ckpt_path  = os.path.join(WORK_DIR, "checkpoint_latest.pth")
    best_path  = os.path.join(WORK_DIR, "best_model_e2e.pth")
    log_path   = os.path.join(WORK_DIR, "training_log_e2e.csv")

    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_srcc   = ckpt['best_val_mos_srcc']
        no_improve  = ckpt.get('epochs_no_improve', 0)
        if os.path.exists(log_path):
            history = pd.read_csv(log_path).to_dict('records')

    print("Starting Training (Dropout=0.3 | Patience=20)...\n")
    for epoch in range(start_epoch, 51):
        lr_bb  = optimizer.param_groups[0]['lr']
        lr_mlp = optimizer.param_groups[1]['lr']
        print(f"\n[Epoch {epoch}/50] LR_BB={lr_bb:.2e} | LR_MLP={lr_mlp:.2e}")
        train_m = run_epoch(model, train_dl, optimizer, device, is_train=True)
        val_m   = run_epoch(model, val_dl,   optimizer, device, is_train=False)
        scheduler.step()

        print(f"\n{'='*68}\nEPOCH {epoch}/50")
        print(f"{'-'*68}")
        print(f"{'Metric':<30} | {'Train':^14} | {'Val':^14}")
        print(f"{'-'*68}")
        for k in ['total_loss','mos_rmse','mos_plcc','mos_srcc','quality_class_rmse'] + \
                 [k for k in train_m if '_f1' in k]:
            print(f"{k:<30} | {train_m[k]:^14.4f} | {val_m[k]:^14.4f}")
        print(f"{'='*68}")

        if val_m['mos_srcc'] > best_srcc:
            print(f"*** New Best SRCC: {val_m['mos_srcc']:.4f} — saving ***")
            best_srcc  = val_m['mos_srcc']
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            print(f"No improvement {no_improve}/{patience}. Best: {best_srcc:.4f}")

        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'best_val_mos_srcc':best_srcc,'epochs_no_improve':no_improve}, ckpt_path)

        row = {'epoch': epoch}
        for k,v in train_m.items(): row[f'train_{k}'] = v
        for k,v in val_m.items():   row[f'val_{k}']   = v
        history.append(row)
        pd.DataFrame(history).to_csv(log_path, index=False)

        if no_improve >= patience:
            print(f"\n[Early Stop] Triggered at epoch {epoch} (patience={patience}).")
            break

    print(f"\nTraining Done! Best Val SRCC: {best_srcc:.4f}")
    print(f"Log: {log_path}")

if __name__ == "__main__":
    main()
