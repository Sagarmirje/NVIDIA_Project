import torch
import torch.nn as nn

class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim=1280):
        super(MultiHeadMLP, self).__init__()
        
        # ──────────────────────────────────────────
        # Shared Trunk
        # ──────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1), # Changed from 0.3
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1) # Changed from 0.2
        )
        
        # ──────────────────────────────────────────
        # Regression Heads 
        # ──────────────────────────────────────────
        self.head_mos = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.head_quality_class = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # ──────────────────────────────────────────
        # Binary Classification Heads
        # ──────────────────────────────────────────
        self.head_spatial = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_hallucination = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_lighting = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_rendering = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_physics = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_flicker = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.head_motion = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_features = self.trunk(x)
        
        outputs = {
            'mos': self.head_mos(shared_features),
            'quality_class': self.head_quality_class(shared_features),
            'spatial_flag': self.head_spatial(shared_features),
            'hallucination_flag': self.head_hallucination(shared_features),
            'lighting_flag': self.head_lighting(shared_features),
            'rendering_flag': self.head_rendering(shared_features),
            'physics_violation_flag': self.head_physics(shared_features),
            'object_flicker_flag': self.head_flicker(shared_features),
            'motion_inconsistency_flag': self.head_motion(shared_features)
        }
        return outputs

def compute_loss(predictions, targets):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # ──────────────────────────────────────────
    # Component Losses
    # ──────────────────────────────────────────
    mos_loss = mse_loss(predictions['mos'], targets['mos'])
    quality_class_loss = mse_loss(predictions['quality_class'], targets['quality_class'])
    
    spatial_flag_loss = bce_loss(predictions['spatial_flag'], targets['spatial_flag'])
    hallucination_flag_loss = bce_loss(predictions['hallucination_flag'], targets['hallucination_flag'])
    lighting_flag_loss = bce_loss(predictions['lighting_flag'], targets['lighting_flag'])
    rendering_flag_loss = bce_loss(predictions['rendering_flag'], targets['rendering_flag'])
    physics_violation_flag_loss = bce_loss(predictions['physics_violation_flag'], targets['physics_violation_flag'])
    object_flicker_flag_loss = bce_loss(predictions['object_flicker_flag'], targets['object_flicker_flag'])
    motion_inconsistency_flag_loss = bce_loss(predictions['motion_inconsistency_flag'], targets['motion_inconsistency_flag'])
    
    # ──────────────────────────────────────────
    # Weighted Total Loss (Rebalanced)
    # ──────────────────────────────────────────
    total_loss = (0.4 * mos_loss) \
               + (0.2 * quality_class_loss) \
               + (0.4 * spatial_flag_loss) \
               + (0.4 * hallucination_flag_loss) \
               + (0.4 * lighting_flag_loss) \
               + (0.4 * rendering_flag_loss) \
               + (0.4 * physics_violation_flag_loss) \
               + (0.4 * object_flicker_flag_loss) \
               + (0.4 * motion_inconsistency_flag_loss)
               
    loss_dict = {
        'mos_loss': mos_loss.item(),
        'quality_class_loss': quality_class_loss.item(),
        'spatial_flag_loss': spatial_flag_loss.item(),
        'hallucination_flag_loss': hallucination_flag_loss.item(),
        'lighting_flag_loss': lighting_flag_loss.item(),
        'rendering_flag_loss': rendering_flag_loss.item(),
        'physics_violation_flag_loss': physics_violation_flag_loss.item(),
        'object_flicker_flag_loss': object_flicker_flag_loss.item(),
        'motion_inconsistency_flag_loss': motion_inconsistency_flag_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict
