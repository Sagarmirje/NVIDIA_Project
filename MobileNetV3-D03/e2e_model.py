import torch
import torch.nn as nn
import torchvision.models as models


class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x    = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out  = torch.zeros_like(x)
        out[:, 1:,   :fold]       = x[:, :-1, :fold]
        out[:, :-1,  fold:2*fold] = x[:, 1:,  fold:2*fold]
        out[:, :,    2*fold:]     = x[:, :,    2*fold:]
        return out.view(nt, c, h, w)


def _inject_tsm(block, n_segment=8):
    if not hasattr(block, 'block'):
        return
    tsm    = TemporalShift(n_segment=n_segment)
    layers = [tsm] + list(block.block.children())
    block.block = nn.Sequential(*layers)


class EndToEndVQAModel(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        self.backbone_features = base.features

        # 960 → 1280 projection initialised from pretrained classifier[0]
        self.proj_960_to_1280 = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish()
        )
        with torch.no_grad():
            self.proj_960_to_1280[0].weight.copy_(base.classifier[0].weight)
            self.proj_960_to_1280[0].bias.copy_(base.classifier[0].bias)

        # Inject TSM into every InvertedResidual block
        for idx in range(len(self.backbone_features)):
            _inject_tsm(self.backbone_features[idx], n_segment=8)

        # Freeze all first
        for param in self.backbone_features.parameters():
            param.requires_grad = False
        for param in self.proj_960_to_1280.parameters():
            param.requires_grad = False

        # Unfreeze last 2 InvRes + final Conv + projection
        for idx in [14, 15, 16]:
            for param in self.backbone_features[idx].parameters():
                param.requires_grad = True
        for param in self.proj_960_to_1280.parameters():
            param.requires_grad = True

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── MLP Head (Dual Trunk) — Dropout=0.3 ────────────────
        self.shared_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

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
        x        = x.view(B * T, C, H, W)
        features = self.backbone_features(x)        # [B*T, 960, 7, 7]
        features = self.gap(features)               # [B*T, 960, 1, 1]
        features = features.view(B * T, -1)         # [B*T, 960]
        features = self.proj_960_to_1280(features)  # [B*T, 1280]
        features = features.view(B, T, -1)          # [B, T, 1280]
        features = features.mean(dim=1)             # [B, 1280]

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

    total_loss = (0.4 * mse_loss(predictions['mos'],           targets['mos'])
                + 0.2 * mse_loss(predictions['quality_class'], targets['quality_class'])
                + 0.4 * bce_loss(predictions['spatial_flag'],             targets['spatial_flag'])
                + 0.4 * bce_loss(predictions['hallucination_flag'],       targets['hallucination_flag'])
                + 0.4 * bce_loss(predictions['lighting_flag'],            targets['lighting_flag'])
                + 0.4 * bce_loss(predictions['rendering_flag'],           targets['rendering_flag'])
                + 0.4 * bce_loss(predictions['physics_violation_flag'],   targets['physics_violation_flag'])
                + 0.4 * bce_loss(predictions['object_flicker_flag'],      targets['object_flicker_flag'])
                + 0.4 * bce_loss(predictions['motion_inconsistency_flag'],targets['motion_inconsistency_flag']))

    return total_loss, {}
