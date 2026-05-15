import torch
import torch.nn as nn
from mmaction.models.backbones.mobilenet_v2_tsm import MobileNetV2TSM

class EndToEndVQAModel(nn.Module):
    def __init__(self):
        super(EndToEndVQAModel, self).__init__()

        # ── Backbone (MobileNetV2-TSM) ──────────────────────────
        self.backbone = MobileNetV2TSM(pretrained='mmcls://mobilenet_v2', num_segments=8, is_shift=True)
        self.backbone.init_weights()

        # Freeze all except last 2 InvertedResidual blocks + conv2
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            if 'layer7' in name or 'layer6.2' in name or 'conv2' in name:
                param.requires_grad = True

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── MLP Head (Dual Trunk) — Dropout=0.3 ────────────────
        self.shared_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Trunk A (Quality)
        self.trunk_a = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),          # <-- changed from 0.1 to 0.3
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head_mos           = nn.Linear(128, 1)
        self.head_quality_class = nn.Linear(128, 1)

        # Trunk B (Artifacts)
        self.trunk_b = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),          # <-- changed from 0.1 to 0.3
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head_spatial       = nn.Linear(128, 1)
        self.head_hallucination = nn.Linear(128, 1)
        self.head_lighting      = nn.Linear(128, 1)
        self.head_rendering     = nn.Linear(128, 1)
        self.head_physics       = nn.Linear(128, 1)
        self.head_flicker       = nn.Linear(128, 1)
        self.head_motion        = nn.Linear(128, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[-1]

        features = self.gap(features)       # [B*T, 1280, 1, 1]
        features = features.view(B, T, -1)  # [B, T, 1280]
        features = features.mean(dim=1)     # [B, 1280]

        shared       = self.shared_proj(features)
        trunk_a_feat = self.trunk_a(shared)
        trunk_b_feat = self.trunk_b(shared)

        return {
            'mos':                      self.head_mos(trunk_a_feat),
            'quality_class':            self.head_quality_class(trunk_a_feat),
            'spatial_flag':             self.head_spatial(trunk_b_feat),
            'hallucination_flag':       self.head_hallucination(trunk_b_feat),
            'lighting_flag':            self.head_lighting(trunk_b_feat),
            'rendering_flag':           self.head_rendering(trunk_b_feat),
            'physics_violation_flag':   self.head_physics(trunk_b_feat),
            'object_flicker_flag':      self.head_flicker(trunk_b_feat),
            'motion_inconsistency_flag':self.head_motion(trunk_b_feat)
        }


def compute_loss(predictions, targets):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    mos_loss           = mse_loss(predictions['mos'],           targets['mos'])
    quality_class_loss = mse_loss(predictions['quality_class'], targets['quality_class'])

    total_loss = (0.4 * mos_loss
                + 0.2 * quality_class_loss
                + 0.4 * bce_loss(predictions['spatial_flag'],             targets['spatial_flag'])
                + 0.4 * bce_loss(predictions['hallucination_flag'],       targets['hallucination_flag'])
                + 0.4 * bce_loss(predictions['lighting_flag'],            targets['lighting_flag'])
                + 0.4 * bce_loss(predictions['rendering_flag'],           targets['rendering_flag'])
                + 0.4 * bce_loss(predictions['physics_violation_flag'],   targets['physics_violation_flag'])
                + 0.4 * bce_loss(predictions['object_flicker_flag'],      targets['object_flicker_flag'])
                + 0.4 * bce_loss(predictions['motion_inconsistency_flag'],targets['motion_inconsistency_flag']))

    return total_loss, {}
