import torch
from ultralytics.nn.modules.block import C3AW_MLM
m = C3AW_MLM(256, 256, e=1.0).eval()
x = torch.randn(1, 256, 25, 25)  # 用奇数尺寸专门卡边界
with torch.no_grad():
    y = m(x)
print(x.shape, y.shape)  # 期待 torch.Size([1,256,25,25]) torch.Size([1,256,25,25])
