import os
import torch
import warnings
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def preprocess_video(video_path):
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError("Video has no frames.")
        
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        
    except Exception:
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        cap.release()
        
        total_frames = len(all_frames)
        if total_frames == 0:
            return torch.zeros((8, 3, 224, 224))
            
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = np.stack([all_frames[i] for i in indices])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    processed_frames = []
    for i in range(8):
        frame = frames[i]
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))
        processed_frames.append(frame)
        
    tensor_frames = torch.tensor(np.stack(processed_frames, axis=0), dtype=torch.float32)
    return tensor_frames

class VideoQADatasetE2E(Dataset):
    def __init__(self, csv_path, videos_dir):
        df_raw = pd.read_csv(csv_path)
        self.videos_dir = videos_dir
        
        self.quality_mapping = {
            'bad': 1.0,
            'poor': 2.0,
            'fair': 3.0,
            'good': 4.0,
            'excellent': 5.0
        }
        
        self.valid_data = []
        video_col = 'filename'
        for col in ['filename', 'video_name', 'video', 'name']:
            if col in df_raw.columns:
                video_col = col
                break
                
        for idx, row in df_raw.iterrows():
            vid = str(row[video_col])
            vid_path = os.path.join(self.videos_dir, vid)
            if not os.path.exists(vid_path):
                alt_vid = vid.replace("_standardized", "")
                vid_path = os.path.join(self.videos_dir, alt_vid)
                
            if os.path.exists(vid_path):
                self.valid_data.append((row, vid_path))

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        row, vid_path = self.valid_data[idx]
        
        video_tensor = preprocess_video(vid_path)
        
        mos = float(row['mos']) / 100.0
        
        qc_raw = row['quality_class']
        if isinstance(qc_raw, str):
            quality_class = self.quality_mapping.get(qc_raw.strip().lower(), 0.0) / 5.0
        else:
            quality_class = float(qc_raw) / 5.0
            
        spatial      = float(row['spatial_flag'])
        hallucination= float(row['hallucination_flag'])
        lighting     = float(row['lighting_flag'])
        rendering    = float(row['rendering_flag'])
        physics      = float(row['physics_violation_flag'])
        flicker      = float(row['object_flicker_flag'])
        motion       = float(row['motion_inconsistency_flag'])
        
        targets = {
            'mos':                      torch.tensor([mos],          dtype=torch.float32),
            'quality_class':            torch.tensor([quality_class],dtype=torch.float32),
            'spatial_flag':             torch.tensor([spatial],      dtype=torch.float32),
            'hallucination_flag':       torch.tensor([hallucination],dtype=torch.float32),
            'lighting_flag':            torch.tensor([lighting],     dtype=torch.float32),
            'rendering_flag':           torch.tensor([rendering],    dtype=torch.float32),
            'physics_violation_flag':   torch.tensor([physics],      dtype=torch.float32),
            'object_flicker_flag':      torch.tensor([flicker],      dtype=torch.float32),
            'motion_inconsistency_flag':torch.tensor([motion],       dtype=torch.float32)
        }
        
        return video_tensor, targets

def get_dataloaders_e2e(splits_dir, videos_dir, batch_size=16):
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv   = os.path.join(splits_dir, "val.csv")
    test_csv  = os.path.join(splits_dir, "test.csv")
    
    train_ds = VideoQADatasetE2E(train_csv, videos_dir)
    val_ds   = VideoQADatasetE2E(val_csv,   videos_dir)
    test_ds  = VideoQADatasetE2E(test_csv,  videos_dir)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader
