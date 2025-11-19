import torch.nn as nn
from .lsnet import LSNet            # ← 刚放进来的文件
from ultralytics.nn.modules.conv import Conv  # YOLO 自带 1×1 Conv

class LSNetBackbone(nn.Module):
    """
    返回 3 个尺度特征：P3(1/8)、P4(1/16)、P5(1/32)，通道适配成 128/256/512
    """
    def __init__(self, variant='t', pretrained=False):
        super().__init__()
        # ① 按 variant 选择宽度
        if variant == 't':
            embed = [64, 128, 256, 384]   # 与 lsnet_t 对应
        elif variant == 's':
            embed = [96, 192, 320, 448]
        elif variant == 'b':
            embed = [128, 256, 384, 512]
        else:
            raise ValueError('variant must be t/s/b')

        # ② 实例化 LSNet 主干（去掉分类头）
        self.net = LSNet(num_classes=0, embed_dim=embed, depth=[0,2,8,10])

        # ③ 1×1 Conv 把通道压到 YOLO 习惯值
        self.adapt = nn.ModuleList([
            Conv(embed[1], 128, 1),   # P3  → 128
            Conv(embed[2], 256, 1),   # P4  → 256
            Conv(embed[3], 512, 1),   # P5  → 512
        ])

    def forward(self, x):
        # patch_embed:  /4
        x = self.net.patch_embed(x)

        # Stage1 (blocks1)  stride /4  → 不用
        x = self.net.blocks1(x)

        # Stage2 stride /8
        x = self.net.blocks2(x)
        p3 = self.adapt[0](x)         # 1/8 输出

        # Stage3 stride /16
        x = self.net.blocks3(x)
        p4 = self.adapt[1](x)         # 1/16

        # Stage4 stride /32
        x = self.net.blocks4(x)
        p5 = self.adapt[2](x)         # 1/32

        return [p3, p4, p5]
