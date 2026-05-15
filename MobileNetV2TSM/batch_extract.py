import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2

# Import the TSM model setup from the previous step
from extract_tsm_features import load_backbone

class VideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths
        # Standard ImageNet means and stds
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        try:
            # Attempt to use decord for efficient reading
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                if total_frames == 0:
                    raise ValueError("No frames")
                indices = np.linspace(0, total_frames - 1, 8, dtype=int)
                frames = vr.get_batch(indices).asnumpy()
            except Exception:
                # Fallback to OpenCV if decord fails or is missing
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
                    raise ValueError("No frames")
                indices = np.linspace(0, total_frames - 1, 8, dtype=int)
                frames = np.stack([all_frames[i] for i in indices])

            # Preprocess the 8 uniformly sampled frames
            processed_frames = []
            for i in range(8):
                frame = frames[i]
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - self.mean) / self.std
                # Convert HWC to CHW
                frame = np.transpose(frame, (2, 0, 1))
                processed_frames.append(frame)
                
            tensor_frames = torch.tensor(np.stack(processed_frames, axis=0), dtype=torch.float32)
            # Return tensor [8, 3, 224, 224], the base filename, and absolute path
            return tensor_frames, base_name, video_path
            
        except Exception as e:
            # If processing fails (e.g., corrupt video), return None
            return None, base_name, video_path

def collate_fn(batch):
    """
    Custom collate function to handle any potentially failed video reads gracefully.
    """
    valid_batch = [b for b in batch if b[0] is not None]
    failed_batch = [b for b in batch if b[0] is None]
    
    failed_paths = [b[2] for b in failed_batch]
    
    if len(valid_batch) == 0:
        return None, [], failed_paths
        
    # Stack tensors into a single batch of shape [B, 8, 3, 224, 224]
    tensors = torch.stack([b[0] for b in valid_batch])
    names = [b[1] for b in valid_batch]
    
    return tensors, names, failed_paths

def main():
    # Environment variables/directories
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    output_dir = "/home/drive3/10k_MobileNet_tsm/extracted_features"
    failed_log = os.path.join(output_dir, "failed_videos.txt")
    
    # 1 & 2. Collect all unique video filenames across train, val, and test splits
    unique_videos = set()
    csv_files = [f for f in os.listdir(splits_dir) if f.endswith('.csv')]
    for csv_file in csv_files:
        csv_path = os.path.join(splits_dir, csv_file)
        df = pd.read_csv(csv_path)
        
        # Locate the filename column
        video_col = None
        for col in ['filename', 'video', 'video_name', 'file_name', 'name']:
            if col in df.columns:
                video_col = col
                break
                
        if video_col:
            for vid in df[video_col]:
                # Standardize mismatch based on prior checks
                alt_vid = str(vid).replace("_standardized", "")
                unique_videos.add(alt_vid)
                
    print(f"Total unique videos collected from CSV splits: {len(unique_videos)}")
    
    # Validate files exist
    video_paths = []
    for vid in unique_videos:
        vid_path = os.path.join(videos_dir, vid)
        if os.path.exists(vid_path):
            video_paths.append(vid_path)
            
    print(f"Total actual videos located on disk: {len(video_paths)}")
    
    # 6. Skip videos that already have a saved .pt file
    to_process = []
    skipped = 0
    for vp in video_paths:
        base_name = os.path.splitext(os.path.basename(vp))[0]
        out_pt = os.path.join(output_dir, f"{base_name}.pt")
        if os.path.exists(out_pt):
            skipped += 1
        else:
            to_process.append(vp)
            
    print(f"Skipping (already extracted): {skipped} videos")
    print(f"Remaining to process: {len(to_process)} videos")
    
    if len(to_process) == 0:
        print("\nAll videos have already been processed.")
        print(f"Total Extracted: 0, Total Skipped: {skipped}, Total Failed: 0")
        return

    # 4. Use a DataLoader with batch processing
    dataset = VideoDataset(to_process)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn, shuffle=False)
    
    # Load model (From Step 2; automatically uses nn.DataParallel for multi-GPU)
    model, device = load_backbone()
    
    extracted_count = 0
    failed_count = 0
    processed_count = 0
    total_target = len(to_process)
    
    print("\nStarting batched feature extraction...")
    
    # Open log for failed videos
    with open(failed_log, "a") as f_log:
        with torch.no_grad():
            for tensors, names, failed_paths in dataloader:
                batch_size_actual = len(names) + len(failed_paths)
                
                # 7. Log failed videos
                for fp in failed_paths:
                    f_log.write(f"{fp}\n")
                    failed_count += 1
                    
                # Extract and save features
                if tensors is not None:
                    tensors = tensors.to(device)
                    # Forward pass through the backbone (yields shape [B, Feature_Dim])
                    features = model(tensors)
                    
                    # Save each feature vector individually
                    for i in range(len(names)):
                        # Create independent 1D tensor clone explicitly moved to CPU
                        feat_1d = features[i].cpu().clone() 
                        out_pt = os.path.join(output_dir, f"{names[i]}.pt")
                        torch.save(feat_1d, out_pt)
                        extracted_count += 1
                
                # 8. Print progress every 100 videos
                prev_count = processed_count
                processed_count += batch_size_actual
                if (processed_count // 100) > (prev_count // 100) or processed_count == total_target:
                    print(f"Processed {processed_count}/{total_target} videos")

    # 9. At the end, print total extracted, total skipped, total failed
    print("\n" + "="*40)
    print("--- Extraction Summary ---")
    print("="*40)
    print(f"Total Extracted: {extracted_count}")
    print(f"Total Skipped:   {skipped}")
    print(f"Total Failed:    {failed_count}")

if __name__ == "__main__":
    main()
