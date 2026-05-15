import torch
from mmaction.models.backbones.mobilenet_v2_tsm import MobileNetV2TSM
m = MobileNetV2TSM(pretrained='mmcls://mobilenet_v2', num_segments=8, is_shift=True)
for name, param in m.named_parameters():
    print(name)
