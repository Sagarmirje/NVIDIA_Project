import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class VQADataset(Dataset):
    def __init__(self, csv_path, features_dir):
        """
        Args:
            csv_path (str): Path to the CSV split (e.g., train.csv, val.csv, test.csv)
            features_dir (str): Directory where the .pt feature files are stored
        """
        self.df = pd.read_csv(csv_path)
        self.features_dir = features_dir
        
        # Mapping categorical 'quality_class' strings to numerical continuous values
        # Typical 5-point Absolute Category Rating (ACR) scale
        self.quality_mapping = {
            'bad': 1.0,
            'poor': 2.0,
            'fair': 3.0,
            'good': 4.0,
            'excellent': 5.0
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ──────────────────────────────────────────
        # 1. Load Pre-extracted Feature Vector
        # ──────────────────────────────────────────
        # Get filename and strip the mismatched suffix just like in extraction
        filename = row.get('filename', row.get('video_name', row.get('video')))
        alt_filename = str(filename).replace("_standardized", "")
        base_name = os.path.splitext(alt_filename)[0]
        
        pt_path = os.path.join(self.features_dir, f"{base_name}.pt")
        
        try:
            # We expect a 1D tensor of shape [1280]
            features = torch.load(pt_path, map_location='cpu', weights_only=True)
        except FileNotFoundError:
            # Fallback to zero-tensor if for some reason a file was skipped/corrupt
            features = torch.zeros(1280)
            
        features = features.float()

        # ──────────────────────────────────────────
        # 2. Extract & Format Targets
        # ──────────────────────────────────────────
        # Regression Targets
        mos = float(row['mos'])
        
        qc_raw = row['quality_class']
        if isinstance(qc_raw, str):
            # Clean string and map safely
            qc_clean = qc_raw.strip().lower()
            quality_class = self.quality_mapping.get(qc_clean, 0.0)
        else:
            quality_class = float(qc_raw)

        # Binary Flags (Pandas Boolean True/False casts natively to 1.0/0.0 float)
        spatial = float(row['spatial_flag'])
        hallucination = float(row['hallucination_flag'])
        lighting = float(row['lighting_flag'])
        rendering = float(row['rendering_flag'])
        physics = float(row['physics_violation_flag'])
        flicker = float(row['object_flicker_flag'])
        motion = float(row['motion_inconsistency_flag'])
        
        # Store in dict natively shaped for loss calculation [1] (becomes [B, 1] after DataLoader batching)
        targets = {
            'mos': torch.tensor([mos], dtype=torch.float32),
            'quality_class': torch.tensor([quality_class], dtype=torch.float32),
            'spatial_flag': torch.tensor([spatial], dtype=torch.float32),
            'hallucination_flag': torch.tensor([hallucination], dtype=torch.float32),
            'lighting_flag': torch.tensor([lighting], dtype=torch.float32),
            'rendering_flag': torch.tensor([rendering], dtype=torch.float32),
            'physics_violation_flag': torch.tensor([physics], dtype=torch.float32),
            'object_flicker_flag': torch.tensor([flicker], dtype=torch.float32),
            'motion_inconsistency_flag': torch.tensor([motion], dtype=torch.float32)
        }
        
        return features, targets

def get_dataloaders(splits_dir, features_dir, batch_size=256, num_workers=4):
    """
    Returns configured train, val, and test DataLoaders.
    """
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_dataset = VQADataset(train_csv, features_dir)
    val_dataset = VQADataset(val_csv, features_dir)
    test_dataset = VQADataset(test_csv, features_dir)
    
    # Only shuffle the training set. Drop last batch on train if desired, but not strictly necessary.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("--- Sanity Check: Dataset & DataLoaders ---")
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    features_dir = "/home/drive3/10k_MobileNet_tsm"
    
    # We will just use a batch size of 8 to test the output dicts map to [8, 1] exactly
    train_dl, val_dl, test_dl = get_dataloaders(splits_dir, features_dir, batch_size=8, num_workers=0)
    
    print(f"Total Train batches: {len(train_dl)}")
    print(f"Total Val batches:   {len(val_dl)}")
    print(f"Total Test batches:  {len(test_dl)}")
    
    # Pull one batch
    features, targets = next(iter(train_dl))
    print(f"\n=> Feature batch shape: {features.shape}")
    print("=> Targets shapes:")
    for k, v in targets.items():
        print(f"     {k:30} -> {v.shape}")
        
    print("\nSanity Check Successful!")
