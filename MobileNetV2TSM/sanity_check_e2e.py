import torch
import torch.nn as nn
from e2e_model import EndToEndVQAModel, compute_loss
from vqa_dataset_e2e import get_dataloaders_e2e

def sanity_check():
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    
    print("1. Initializing Dataset...")
    train_dl, _, _ = get_dataloaders_e2e(splits_dir, videos_dir, batch_size=4)
    
    print("2. Fetching One Batch...")
    videos, targets = next(iter(train_dl))
    print(f"   Video Batch Shape: {videos.shape}")
    for k, v in targets.items():
        print(f"   Target '{k}' Shape: {v.shape}")
        
    print("\n3. Initializing End-to-End Model...")
    model = EndToEndVQAModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    videos = videos.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}
    
    print("\n4. Parameter Freeze Check:")
    frozen = 0
    trainable = 0
    trainable_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
            if 'backbone' in name:
                trainable_names.append(name)
        else:
            frozen += param.numel()
            
    print("   Unfrozen Backbone Layers:")
    prefixes = list(set([".".join(n.split(".")[:3]) for n in trainable_names]))
    for p in sorted(prefixes):
        print(f"     - {p}.*")
        
    print(f"\n   Total Frozen Parameters: {frozen:,}")
    print(f"   Total Trainable Parameters: {trainable:,}")
    
    print("\n5. Forward Pass...")
    outputs = model(videos)
    for k, v in outputs.items():
        print(f"   Output '{k}' Shape: {v.shape}")
        
    print("\n6. Loss Computation & Backward Pass...")
    total_loss, loss_dict = compute_loss(outputs, targets)
    print(f"   Total Loss: {total_loss.item():.4f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    print("   Backward pass successful!")
    if torch.isnan(total_loss) or total_loss.item() > 100:
        print("   WARNING: Loss is abnormal (NaN or > 100).")
    else:
        print("   Loss is reasonable.")
        
if __name__ == "__main__":
    sanity_check()
