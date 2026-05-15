import torch
import torch.nn as nn
import numpy as np
import cv2
from mmaction.models.backbones.mobilenet_v2_tsm import MobileNetV2TSM

class TSMFeatureExtractor(nn.Module):
    """
    A wrapper module to handle the BxT reshaping before passing to the TSM backbone.
    This ensures nn.DataParallel splits along the Batch dimension correctly, instead of
    splitting the Temporal dimension which causes shape mismatches in TSM shift operations.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x is [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # Reshape to [B*T, C, H, W] for the backbone
        x = x.view(B * T, C, H, W)
        
        # Pass through the TSM backbone
        features = self.backbone(x)
        
        # MobileNetV2 returns a tuple of feature maps; get the last one
        if isinstance(features, tuple):
            features = features[-1]  # [B*T, 1280, 7, 7]
            
        # Global Average Pooling
        features = self.gap(features)  # [B*T, C, 1, 1]
        
        # Reshape to [B, T, C]
        features = features.view(B, T, -1)
        
        # Average across the temporal dimension to get [B, C] 1D features per video
        features = features.mean(dim=1)
        
        return features

def load_backbone():
    """
    2. Loads the backbone with pretrained ImageNet weights, removes the classification head,
       and sets it to eval() mode.
    """
    print("Initializing MobileNetV2-TSM backbone...")
    # MMAction2 backbones inherently do not include the classification head (only output features)
    backbone = MobileNetV2TSM(pretrained='mmcls://mobilenet_v2', num_segments=8, is_shift=True)
    backbone.init_weights()
    
    # Wrap in our feature extractor to handle B, T, C, H, W correctly for DataParallel
    model = TSMFeatureExtractor(backbone)
    
    # 6. Support multi-GPU using torch.nn.DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_video(video_path):
    """
    3. Reads video, uniformly samples 8 frames, resizes to 224x224,
       normalizes with ImageNet stats, and returns tensor of shape [1, 8, 3, 224, 224].
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError("Video has no frames.")
        
        # Uniformly sample 8 frames
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # [8, H, W, 3]
        
    except ImportError:
        print("Decord not found, falling back to OpenCV for video reading.")
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
            raise ValueError("Video has no frames.")
            
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = np.stack([all_frames[i] for i in indices])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    processed_frames = []
    for i in range(8):
        frame = frames[i]
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))
        processed_frames.append(frame)
        
    tensor_frames = torch.tensor(np.stack(processed_frames, axis=0), dtype=torch.float32)
    tensor_frames = tensor_frames.unsqueeze(0)  # [1, 8, 3, 224, 224]
    
    return tensor_frames

def extract_features(video_path, model, device):
    """
    4. Preprocesses the video, passes through TSM backbone,
       applies global average pooling on spatial dimensions, 
       and returns a 1D feature vector (flattened).
    """
    video_tensor = preprocess_video(video_path)  # [1, 8, 3, 224, 224]
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        # Pass through the DataParallel model wrapper
        features = model(video_tensor) # [B, C] => [1, 1280]
        
        # Flatten to 1D feature vector
        features_1d = features.flatten()
        
    return features_1d

if __name__ == "__main__":
    # 5. Test on a single sample video and print output shape
    sample_video = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset/00000_07.mp4"
    
    print("Loading model...")
    model, device = load_backbone()
    
    print(f"Extracting features from: {sample_video}")
    try:
        feat_vector = extract_features(sample_video, model, device)
        print("Feature extraction successful!")
        print(f"Output Feature Vector Shape: {feat_vector.shape}")
        print(f"Sample Feature Values (first 5): {feat_vector[:5].cpu().numpy()}")
    except Exception as e:
        print(f"Error during feature extraction: {e}")
