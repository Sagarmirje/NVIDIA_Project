import torch
import torch.nn as nn
import torchvision.models as models

# ──────────────────────────────────────────────────────────────
# Temporal Shift Module (TSM) — identical to the V2 pipeline
# ──────────────────────────────────────────────────────────────
class TemporalShift(nn.Module):
    """
    In-place temporal shift. Shifts 1/8 of channels backward in time
    and 1/8 forward in time so 2-D conv gains temporal awareness.
    The input must arrive as [B*T, C, H, W] with T folded into B.
    """
    def __init__(self, n_segment=8, fold_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div  = fold_div

    def forward(self, x):
        nt, c, h, w  = x.size()
        n_batch      = nt // self.n_segment
        x            = x.view(n_batch, self.n_segment, c, h, w)
        fold         = c // self.fold_div

        out          = torch.zeros_like(x)
        # Shift backward (past → present)
        out[:, 1:,    :fold]   = x[:, :-1, :fold]
        # Shift forward  (future → present)
        out[:, :-1,   fold:2*fold] = x[:, 1:, fold:2*fold]
        # Keep the rest unchanged
        out[:, :,     2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)


def _inject_tsm_into_inverted_residual(block, n_segment=8):
    """
    Prepend a TemporalShift before the first depthwise conv inside an
    InvertedResidual block (works for both V2 and V3 variants from
    torchvision because they all expose a `.block` Sequential).
    """
    if not hasattr(block, 'block'):
        return  # not an InvertedResidual — skip safely

    tsm   = TemporalShift(n_segment=n_segment)
    inner = block.block

    # inner is a Sequential; we insert TSM at position 0
    layers = [tsm] + list(inner.children())
    block.block = nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────
# End-to-End VQA Model — MobileNetV3-Large + TSM backbone
# ──────────────────────────────────────────────────────────────
class EndToEndVQAModel(nn.Module):
    """
    Same Dual-Trunk Multi-Head MLP as the V2 pipeline.
    Only Stage 3 (backbone) is swapped to MobileNetV3-Large-TSM.

    MobileNetV3-Large feature dims:
        features output  →  [B*T, 960, 7, 7]
        after GAP+flatten → [B,   960]
        after proj 960→1280 → [B, 1280]   ← identical to V2 output dim
    """

    N_SEGMENTS = 8

    def __init__(self):
        super(EndToEndVQAModel, self).__init__()

        # ── Load ImageNet-pretrained MobileNetV3-Large ──────────
        base = models.mobilenet_v3_large(weights='IMAGENET1K_V1')

        # features[0..16]: Conv2dNormActivation + 15 InvertedResiduals + final Conv
        self.backbone_features = base.features   # nn.Sequential of 17 blocks

        # 960 → 1280 projection (same linear as MobileNetV3's classifier[0])
        # We detach it from the classifier and use it as part of the backbone
        self.proj_960_to_1280 = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish()
        )
        # Initialise with the pretrained weights
        with torch.no_grad():
            self.proj_960_to_1280[0].weight.copy_(base.classifier[0].weight)
            self.proj_960_to_1280[0].bias.copy_(base.classifier[0].bias)

        # ── Inject TSM into every InvertedResidual block ────────
        for idx in range(len(self.backbone_features)):
            block = self.backbone_features[idx]
            _inject_tsm_into_inverted_residual(block, n_segment=self.N_SEGMENTS)

        # ── Freeze strategy (mirror of V2 pipeline) ─────────────
        # Freeze ALL backbone features first
        for param in self.backbone_features.parameters():
            param.requires_grad = False
        for param in self.proj_960_to_1280.parameters():
            param.requires_grad = False

        # Unfreeze last 2 InvertedResidual blocks (indices 14, 15)
        # and the final Conv2dNormActivation (index 16) — equiv. of
        # layer6.2 / layer7.0 / conv2 in the V2 architecture
        for idx in [14, 15, 16]:
            for param in self.backbone_features[idx].parameters():
                param.requires_grad = True
        # Also unfreeze the 960→1280 projection (equiv. of V2 conv2 head)
        for param in self.proj_960_to_1280.parameters():
            param.requires_grad = True

        # ── Global Average Pooling ───────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── MLP Head (Dual Trunk) — identical to V2 pipeline ────
        # Input dim is now 1280 (after 960→1280 projection)
        self.shared_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Trunk A — Quality / Aesthetics
        self.trunk_a = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head_mos           = nn.Linear(128, 1)
        self.head_quality_class = nn.Linear(128, 1)

        # Trunk B — Artifact Detection
        self.trunk_b = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        # x : [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)          # [B*T, C, H, W]

        features = self.backbone_features(x) # [B*T, 960, 7, 7]
        features = self.gap(features)        # [B*T, 960, 1, 1]
        features = features.view(B * T, -1)  # [B*T, 960]
        features = self.proj_960_to_1280(features)  # [B*T, 1280]
        features = features.view(B, T, -1)   # [B, T, 1280]
        features = features.mean(dim=1)      # [B, 1280]

        shared       = self.shared_proj(features)
        trunk_a_feat = self.trunk_a(shared)
        trunk_b_feat = self.trunk_b(shared)

        outputs = {
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
        return outputs


# ── Multi-Task Loss — identical to V2 pipeline ────────────────
def compute_loss(predictions, targets):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    mos_loss           = mse_loss(predictions['mos'],           targets['mos'])
    quality_class_loss = mse_loss(predictions['quality_class'], targets['quality_class'])

    spatial_loss      = bce_loss(predictions['spatial_flag'],             targets['spatial_flag'])
    halluc_loss       = bce_loss(predictions['hallucination_flag'],       targets['hallucination_flag'])
    lighting_loss     = bce_loss(predictions['lighting_flag'],            targets['lighting_flag'])
    rendering_loss    = bce_loss(predictions['rendering_flag'],           targets['rendering_flag'])
    physics_loss      = bce_loss(predictions['physics_violation_flag'],   targets['physics_violation_flag'])
    flicker_loss      = bce_loss(predictions['object_flicker_flag'],      targets['object_flicker_flag'])
    motion_loss       = bce_loss(predictions['motion_inconsistency_flag'],targets['motion_inconsistency_flag'])

    total_loss = (0.4 * mos_loss
                + 0.2 * quality_class_loss
                + 0.4 * spatial_loss
                + 0.4 * halluc_loss
                + 0.4 * lighting_loss
                + 0.4 * rendering_loss
                + 0.4 * physics_loss
                + 0.4 * flicker_loss
                + 0.4 * motion_loss)

    loss_dict = {
        'mos_loss':                      mos_loss.item(),
        'quality_class_loss':            quality_class_loss.item(),
        'spatial_flag_loss':             spatial_loss.item(),
        'hallucination_flag_loss':       halluc_loss.item(),
        'lighting_flag_loss':            lighting_loss.item(),
        'rendering_flag_loss':           rendering_loss.item(),
        'physics_violation_flag_loss':   physics_loss.item(),
        'object_flicker_flag_loss':      flicker_loss.item(),
        'motion_inconsistency_flag_loss':motion_loss.item(),
        'total_loss':                    total_loss.item()
    }

    return total_loss, loss_dict
