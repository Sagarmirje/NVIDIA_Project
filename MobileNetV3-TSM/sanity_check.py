"""
sanity_check.py  — Run ONE forward batch through the full pipeline
to verify shapes and loss before starting full training.
"""
import torch
from e2e_model import EndToEndVQAModel, compute_loss
from vqa_dataset_e2e import get_dataloaders_e2e

def main():
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"

    print("Loading 1 batch for sanity check...")
    train_dl, _, _ = get_dataloaders_e2e(splits_dir, videos_dir, batch_size=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = EndToEndVQAModel().to(device)

    total     = sum(p.numel() for p in model.parameters())
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = total - frozen
    print(f"\nBackbone raw output  : 960-dim (projected to 1280-dim)")
    print(f"Total parameters     : {total:,}")
    print(f"Frozen parameters    : {frozen:,}")
    print(f"Trainable parameters : {trainable:,}")

    videos, targets = next(iter(train_dl))
    videos  = videos.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}

    print(f"\nInput tensor shape   : {videos.shape}")

    with torch.no_grad():
        preds = model(videos)

    print("\nOutput shapes:")
    for k, v in preds.items():
        print(f"  {k:<35}: {v.shape}")

    loss, loss_dict = compute_loss(preds, targets)
    print(f"\nTotal loss           : {loss.item():.4f}")
    print("\nPer-task losses:")
    for k, v in loss_dict.items():
        print(f"  {k:<40}: {v:.4f}")

    print("\n✅  Sanity check PASSED — pipeline is ready for training.")

if __name__ == "__main__":
    main()
