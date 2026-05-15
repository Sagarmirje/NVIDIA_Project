import os
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class VideoQADataset(Dataset):
    def __init__(self, csv_path, features_dir):
        """
        Accepts a CSV filepath and features directory path.
        Filters and skips rows where the .pt feature file does not exist, emitting a warning.
        """
        df_raw = pd.read_csv(csv_path)
        self.features_dir = features_dir
        
        # Categorical string mapping for the 'quality_class' column
        self.quality_mapping = {
            'bad': 1.0,
            'poor': 2.0,
            'fair': 3.0,
            'good': 4.0,
            'excellent': 5.0
        }
        
        self.valid_data = []
        missing_count = 0
        
        # Correctly identify filename column based on Step 1 output
        video_col = 'filename'
        for col in ['filename', 'video_name', 'video', 'name']:
            if col in df_raw.columns:
                video_col = col
                break
                
        # Filter rows to only keep those with existing .pt files
        for idx, row in df_raw.iterrows():
            vid = str(row[video_col]).replace("_standardized", "")
            base_name = os.path.splitext(vid)[0]
            pt_path = os.path.join(self.features_dir, f"{base_name}.pt")
            
            if os.path.exists(pt_path):
                self.valid_data.append((row, pt_path))
            else:
                missing_count += 1
                
        if missing_count > 0:
            warnings.warn(f"[{os.path.basename(csv_path)}] Skipped {missing_count} rows because the .pt feature file does not exist in {self.features_dir}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        row, pt_path = self.valid_data[idx]
        
        # Load feature tensor
        features = torch.load(pt_path, map_location='cpu', weights_only=True)
        features = features.float()
        
        if features.dim() > 1:
            features = features.flatten()
            
        # ──────────────────────────────────────────
        # Normalization constants (Must denormalize during metric tracking!)
        # ──────────────────────────────────────────
        # Normalize mos: divide by 100.0 -> maps to [0, 1]
        mos = float(row['mos']) / 100.0
        
        # Normalize quality_class: divide by 5.0 -> maps to [0, 1]
        qc_raw = row['quality_class']
        if isinstance(qc_raw, str):
            quality_class = self.quality_mapping.get(qc_raw.strip().lower(), 0.0) / 5.0
        else:
            quality_class = float(qc_raw) / 5.0
            
        spatial = float(row['spatial_flag'])
        hallucination = float(row['hallucination_flag'])
        lighting = float(row['lighting_flag'])
        rendering = float(row['rendering_flag'])
        physics = float(row['physics_violation_flag'])
        flicker = float(row['object_flicker_flag'])
        motion = float(row['motion_inconsistency_flag'])
        
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

def get_dataloaders(splits_dir, features_dir, batch_size=64):
    """
    Automatically finds train/val/test CSVs in splits_dir and returns DataLoaders.
    """
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_ds = VideoQADataset(train_csv, features_dir)
    val_ds = VideoQADataset(val_csv, features_dir)
    test_ds = VideoQADataset(test_csv, features_dir)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader
