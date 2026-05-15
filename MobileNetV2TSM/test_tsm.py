import torch
from mmaction.models.backbones.mobilenet_v2_tsm import MobileNetV2TSM

model = MobileNetV2TSM(num_segments=8, is_shift=True, pretrained2d=False)
x = torch.randn(8, 3, 224, 224)
out = model(x)
if isinstance(out, tuple):
    for idx, o in enumerate(out):
        print(f"Output {idx} shape:", o.shape)
else:
    print("Output shape:", out.shape)
