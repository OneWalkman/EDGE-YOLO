# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .conv import Conv, DSConv, DWConv, GhostConv, LightConv, RepConv, autopad,WTConv2d
from .transformer import TransformerBlock
import pywt

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "HyperACE", 
    "DownsampleConv", 
    "FullPAD_Tunnel",
    "DSC3K2",#yolov13 模块
    "RHJM",
    "MulGate",
    "SPPF_Wavelet",
    "HyperACE_Wavelet",
    "DSC3K2_MSLA",
    "MSLA",
    "DSC3K2_LGL",
    "Wavelet_SS2D",
    "C2PSA_LinearAttention",
    "C3k2_Wavelet",
    "DSC3K2_Wavelet",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class MulGate(nn.Module):
    def __init__(self, c, expansion=3, k=7, d=1, gamma=1e-2):
        super().__init__()
        self.pre = DSConv(c, c, k=k, d=d)
        h = int(c * expansion)
        self.f1, self.f2 = nn.Conv2d(c, h, 1), nn.Conv2d(c, h, 1)
        self.act = nn.ReLU6(inplace=True)
        self.mix = nn.Conv2d(h, c, 1, bias=False)
        self.bn  = nn.BatchNorm2d(c)
        self.gamma = nn.Parameter(torch.full((1, c, 1, 1), gamma))
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.bn.weight); nn.init.zeros_(self.bn.bias)
        
    def forward(self, x):
        y = self.pre(x)
        g = self.act(self.f1(y)) * self.f2(y)
        z = self.bn(self.mix(g))
        return x + self.gamma * z


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class HaarDWT2D(nn.Module):
    """
    输入: (B, C, H, W) -> 输出四个子带 (LL, LH, HL, HH)，形状均为 (B, C, H/2, W/2)
    """
    def __init__(self):
        super().__init__()
        # Haar 滤波器
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5],
                           [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5],
                           [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5,  0.5]], dtype=torch.float32)
        # 作为 buffer 保存
        self.register_buffer("ll", ll)
        self.register_buffer("lh", lh)
        self.register_buffer("hl", hl)
        self.register_buffer("hh", hh)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # 组装分组卷积核: (4*C, 1, 2, 2)，每个通道复用同一组 4 个核
        k = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0)  # (4, 2, 2)
        k = k.view(4, 1, 2, 2).repeat(C, 1, 1, 1)                     # (4*C,1,2,2)
        # 分组卷积，步长2实现下采样
        y = F.conv2d(x, k, bias=None, stride=2, padding=0, groups=C)  # (B, 4*C, H/2, W/2)
        # 按通道拆成四个子带
        y = y.view(B, C, 4, H // 2, W // 2)                           # (B, C, 4, H/2, W/2)
        LL = y[:, :, 0, ...]
        LH = y[:, :, 1, ...]
        HL = y[:, :, 2, ...]
        HH = y[:, :, 3, ...]
        return LL, LH, HL, HH

class SPPF_Wavelet(nn.Module):
    """
    Wavelet-Enhanced SPPF (通道自洽版)
    - 签名: (c1, c2, k=5) —— 与 Ultralytics SPPF 保持一致
    - 步骤:
        1) cv1: c1 -> c_ = c1//2
        2) 对 y0 做 Haar DWT 得到 (LL, LH, HL, HH) : 各为 (B, c_, H/2, W/2)
        3) 低频与三个高频分别用 1x1/3x3 卷积处理，通道从 c_ -> c_//2
        4) 上采样回 (H, W)，与 y0 拼接，总通道: c_ + 4*(c_//2) = 3*c_
        5) cv2: 3*c_ -> c2
    """
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        assert k % 2 == 1, "SPPF_Wavelet: kernel size k 必须为奇数"
        self.c1 = c1
        self.c2 = c2
        c_ = c1 // 2
        self.c_ = c_

        # 1) 先降通道
        self.cv1 = Conv(c1, c_, 1, 1)

        # 2) DWT
        self.dwt = HaarDWT2D()  # 若你已有 HaarDWT2D(c_) 的版本，可替换为你的类

        # 3) 子带处理：低频更偏平滑，1x1；高频共享一个 3x3 更聚焦边缘
        self.f_ll = Conv(c_, c_ // 2, 1, 1)
        self.f_h  = Conv(c_, c_ // 2, 3, 1)

        # 4) 最终映射到 c2。拼接通道: y0( c_ ) + LL↑/LH↑/HL↑/HH↑( 4 * c_/2 ) == 3*c_
        self.cv2 = Conv(3 * c_, c2, 1, 1)

    @staticmethod
    def _upsample(x: torch.Tensor, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor):
        # 通道自检（能快速定位上游输出通道是否按 YAML 推导一致）
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"[SPPF_Wavelet] 输入通道不匹配: 期望 {self.c1}, 实际 {x.shape[1]}。\n"
                f"请检查上游模块（例如 DSC3K2）是否将最终输出投影回其 c2（而非中间的 c_=int(c2*e)）。"
            )

        # 1) c1 -> c_
        y0 = self.cv1(x)  # (B, c_, H, W)

        # 2) DWT 四分 (各 (B, c_, H/2, W/2))
        LL, LH, HL, HH = self.dwt(y0)

        # 3) 子带处理 + 上采样回原尺寸
        H, W = y0.shape[-2:]
        LLu = self._upsample(self.f_ll(LL), (H, W))  # (B, c_/2, H, W)
        LHu = self._upsample(self.f_h(LH),  (H, W))  # (B, c_/2, H, W)
        HLu = self._upsample(self.f_h(HL),  (H, W))  # (B, c_/2, H, W)
        HHu = self._upsample(self.f_h(HH),  (H, W))  # (B, c_/2, H, W)

        # 4) 拼接 -> 3*c_
        y = torch.cat([y0, LLu, LHu, HLu, HHu], dim=1)  # (B, 3*c_, H, W)

        # 5) 3*c_ -> c2
        return self.cv2(y)

class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x



class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))

class DSBottleneck(nn.Module):
    """
    An improved bottleneck block using depthwise separable convolutions (DSConv).

    This class implements a lightweight bottleneck module that replaces standard convolutions with depthwise
    separable convolutions to reduce parameters and computational cost. 

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to use a residual shortcut connection. The connection is only added if c1 == c2. Defaults to True.
        e (float, optional): Expansion ratio for the intermediate channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv layer. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv layer. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv layer. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSBottleneck module.

    Examples:
        >>> import torch
        >>> model = DSBottleneck(c1=64, c2=64, shortcut=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)   
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)  
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    """
    An improved C3k module using DSBottleneck blocks for lightweight feature extraction.

    This class extends the C3 module by replacing its standard bottleneck blocks with DSBottleneck blocks,
    which use depthwise separable convolutions.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of DSBottleneck blocks to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections within the DSBottlenecks. Defaults to True.
        g (int, optional): Number of groups for grouped convolution (passed to parent C3). Defaults to 1.
        e (float, optional): Expansion ratio for the C3 module's hidden channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv in each DSBottleneck. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in each DSBottleneck. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv in each DSBottleneck. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k module (inherited from C3).

    Examples:
        >>> import torch
        >>> model = DSC3k(c1=128, c2=128, n=2, k1=3, k2=7)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 64, 64])
    """
    def __init__(
        self,
        c1,                
        c2,                 
        n=1,                
        shortcut=True,      
        g=1,                 
        e=0.5,              
        k1=3,               
        k2=5,               
        d2=1                 
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  

        self.m = nn.Sequential(
            *(
                DSBottleneck(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )

class DSC3K2(C2f):
    """
    An improved C3k2 module that uses lightweight depthwise separable convolution blocks.

    This class redesigns C3k2 module, replacing its internal processing blocks with either DSBottleneck
    or DSC3k modules.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of internal processing blocks to stack. Defaults to 1.
        dsc3k (bool, optional): If True, use DSC3k as the internal block. If False, use DSBottleneck. Defaults to False.
        e (float, optional): Expansion ratio for the C2f module's hidden channels. Defaults to 0.5.
        g (int, optional): Number of groups for grouped convolution (passed to parent C2f). Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections in the internal blocks. Defaults to True.
        k1 (int, optional): Kernel size for the first DSConv in internal blocks. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in internal blocks. Defaults to 7.
        d2 (int, optional): Dilation for the second DSConv in internal blocks. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3K2 module (inherited from C2f).

    Examples:
        >>> import torch
        >>> # Using DSBottleneck as internal block
        >>> model1 = DSC3K2(c1=64, c2=64, n=2, dsc3k=False)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output1 = model1(x)
        >>> print(f"With DSBottleneck: {output1.shape}")
        With DSBottleneck: torch.Size([2, 64, 128, 128])
        >>> # Using DSC3k as internal block
        >>> model2 = DSC3K2(c1=64, c2=64, n=1, dsc3k=True)
        >>> output2 = model2(x)
        >>> print(f"With DSC3k: {output2.shape}")
        With DSC3k: torch.Size([2, 64, 128, 128])
    """
    def __init__(
        self,
        c1,          
        c2,         
        n=1,          
        dsc3k=False,  
        e=0.5,       
        g=1,        
        shortcut=True,
        k1=3,       
        k2=7,       
        d2=1         
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.c, self.c,
                    n=2,           
                    shortcut=shortcut,
                    g=g,
                    e=1.0,  
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(
                    self.c, self.c,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )

class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.

    Examples:
        >>> import torch
        >>> model = AdaHyperedgeGen(node_dim=64, num_hyperedges=16, num_heads=4)
        >>> x = torch.randn(2, 100, 64)  # (Batch, Num_Nodes, Node_Dim)
        >>> A = model(x)
        >>> print(A.shape)
        torch.Size([2, 100, 16])
    """
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both":
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
    
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)           
            max_context, _ = X.max(dim=1)           
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets           
        
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)

class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    Examples:
        >>> import torch
        >>> model = AdaHGConv(embed_dim=128, num_hyperedges=16, num_heads=8)
        >>> x = torch.randn(2, 256, 128) # (Batch, Num_Nodes, Dim)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 256, 128])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X)  
        
        He = torch.bmm(A.transpose(1, 2), X) 
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X
        
class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) 
        tokens = self.hgnn(tokens) 
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out 

class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    Examples:
        >>> import torch
        >>> model = C3AH(c1=64, c2=128, num_hyperedges=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])
    """
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_, 
                          num_hyperedges=num_hyperedges, 
                          num_heads=num_heads,
                          dropout=0.1,
                          context=context)
        self.cv3 = Conv(2 * c_, c2, 1)  
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class FuseModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out

class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
        super().__init__()
        self.c = int(c2 * e1) 
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)
                    
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))

class DownsampleConv(nn.Module):
    """
    A simple downsampling block with optional channel adjustment.

    This module uses average pooling to reduce the spatial dimensions (H, W) by a factor of 2. It can
    optionally include a 1x1 convolution to adjust the number of channels, typically doubling them.

    Attributes:
        in_channels (int): The number of input channels.
        channel_adjust (bool, optional): If True, a 1x1 convolution doubles the channel dimension. Defaults to True.

    Methods:
        forward: Performs the downsampling and optional channel adjustment.

    Examples:
        >>> import torch
        >>> model = DownsampleConv(in_channels=64, channel_adjust=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """
    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
        else:
            self.channel_adjust = nn.Identity() 

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))

class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out


class RHJM(nn.Module):
    def __init__(self, in_ch, local_size=5, gamma=2, b=1, local_weight=0.5):
        super().__init__()
        self.local_size = local_size
        self.local_weight = local_weight

        # ECA 风格核大小：k 为依赖通道数的奇数小核
        t = int(abs(math.log(in_ch, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        k = max(k, 1)  # 兜底：至少为 1

        # 1D 卷积（共享核，超轻量）
        self.conv_global = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local  = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_avg_pool  = nn.AdaptiveAvgPool2d(local_size)  # -> (S,S)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)           # -> (1,1)

    def forward(self, x):                   # x: (B,C,H,W)
        B, C, H, W = x.shape

        # Local: (B,C,H,W) -> (B,C,S,S)
        x_local = self.local_avg_pool(x)    # (B,C,S,S)
        S = self.local_size

        # Global: (B,C,H,W) -> (B,C,1,1)
        x_global = self.global_avg_pool(x)  # 更严谨（直接 from x）

        # ---- Local branch: 序列化(位置×通道) -> 1D 卷积 -> 还原 ----
        # (B,C,S,S) -> (B,1,S*S*C)
        seq_local = x_local.view(B, C, -1).transpose(1, 2).reshape(B, 1, -1).contiguous()
        out_local = self.conv_local(seq_local)                                # (B,1,S*S*C)
        att_local = out_local.view(B, S*S, C).transpose(1, 2)\
                               .reshape(B, C, S, S).sigmoid()                 # (B,C,S,S)

        # ---- Global branch: 通道序列 -> 1D 卷积 -> 广播到 S×S ----
        # (B,C,1,1) -> (B,1,C)
        seq_global = x_global.view(B, C, 1).transpose(1, 2).contiguous()      # (B,1,C)
        out_global = self.conv_global(seq_global).transpose(1, 2)             # (B,C,1)
        att_global = out_global.unsqueeze(-1).sigmoid().expand(B, C, S, S)    # (B,C,S,S)

        # ---- 融合并对齐到 (H,W) ----
        att = att_global * (1 - self.local_weight) + att_local * self.local_weight  # (B,C,S,S)
        att = F.adaptive_avg_pool2d(att, (H, W))                                    # (B,C,H,W)

        return x * att

class HyperACE_Wavelet(nn.Module):
    """
    Wavelet-enhanced
    
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
        super().__init__()
        self.c = int(c2 * e1) 
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AW_MLM(self.c, self.c, e2)
        self.branch2 = C3AW_MLM(self.c, self.c, e2)
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))
    
class Wavelet_SS2D(nn.Module):
    """
    Wavelet-SS2D enhanced.
    
    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
        super().__init__()
        
        self.c = int(c2 * e1) 
        g1 = math.gcd(c1, 3 * self.c) or 1
        g2 = math.gcd((4 + n) * self.c, c2) or 1
        # 可选：限制最大分组，避免过细分导致 BN 统计不稳
        g1 = min(g1, 8)
        g2 = min(g2, 8)

                # __init__
        self.mod_detach = True     # 训练初期可先 True
        r = 4                      # 压缩率
         # 注意：DetectionModel 在 __init__ 里会以 train() 模式跑一次 dummy 前向；
        # GAP 后是 1×1，若这里用带 BN 的 Conv 会在 B=1 时触发 BN 训练态错误。
        # 因此改为**无 BN**的 1×1-MLP。
        self.mod_film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c, self.c // r, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(self.c // r, 2 * self.c, kernel_size=1, bias=True),
        )
        # 初始置零，确保初始调制为恒等（更稳定）
        nn.init.zeros_(self.mod_film[3].weight)
        nn.init.zeros_(self.mod_film[3].bias)

        
        self.cv1 = Conv(c1, 3 * self.c, 1, 1, g=g1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1, 1, g=g2)

        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AW_MLM(self.c, self.c, e2)
        self.branch2 = LocalSS2DContext(self.c, depth=1, step=2, expand=e2, window_size=8, shift=True)
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1], cond=out1.detach())
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        
        # forward 里（在 out1/out2 已计算之后，cat 之前）
        src = out1.detach() if self.mod_detach else out1
        gamma_beta = self.mod_film(src)                  # (B,2C,1,1)
        gamma, beta = gamma_beta.split(self.c, dim=1)    # (B,C,1,1), (B,C,1,1)
        out2 = out2 * (1 + torch.tanh(gamma)) + beta

        y.append(out2)
        return self.cv2(torch.cat(y, 1))

class SS2DContext(nn.Module):
    """
    SS2D global context modeling (JamMa's JEGO-style, single-image adaptation).

    Args:
        c (int): channel dimension.
        depth (int): number of mixer layers per direction (stacked).
        step (int): skip step for scan (default=2 as in JamMa).
        use_mamba (bool): if True and mamba_module.create_block is available, use Mamba; otherwise fallback.
        expand (int): channel expansion in fallback mixer.

    Input:
        x: Tensor (B, c, H, W)

    Output:
        y: Tensor (B, c, H, W)  (same shape as input)
    """
    def __init__(self, c, depth=1, step=2, use_mamba=False, expand=2):
        super().__init__()
        self.c = c
        self.depth = depth
        self.step = step
        self.use_mamba = False
        self.mixers = nn.ModuleList()

        if use_mamba:
            try:
                # Try to import JamMa's create_block (preferred)
                from mamba_module import create_block  # expects to be importable in the project
                self.use_mamba = True
                for i in range(depth * 4):
                    self.mixers.append(create_block(c))
            except Exception:
                self.use_mamba = False

        if not self.use_mamba:
            for i in range(depth * 4):
                self.mixers.append(SeqMixer1D(c, hidden=expand))

        self.agg = GLU2DAggregator(c)

    def forward(self, x):
        B, C, H, W = x.shape
        xs, oh, ow = _scan_jego_single(x, self.step)  # (B,4,L,C)
        # process 4 directional sequences, possibly stacked depth times
        # reshape to a list for convenience: four tensors (B,L,C)
        seqs = [xs[:, i] for i in range(4)]
        for d in range(self.depth):
            for k in range(4):
                mixer = self.mixers[d * 4 + k]
                seqs[k] = mixer(seqs[k])  # (B,L,C)
        y = torch.stack(seqs, dim=1).transpose(2, 3)  # (B,4,C,L)
        y2d = _merge_jego_single(y, oh, ow, self.step)  # (B,C,H,W)
        return self.agg(y2d)
    
# ======= LocalSS2DContext (clean, unified) =======
class LocalSS2DContext(nn.Module):
    """
    Windowed Selective Scan 2D with wavelet-guided directional weights.
    (B, C, H, W) -> (B, C, H, W)

    Features:
      - Wavelet prior (LH/HL) + learnable bias (dir_gate)
      - HH-based uniform compensation
      - w-regularizer: {"none","entropy","sparse"}
    """
    def __init__(self, c, depth=1, step=2, use_mamba=False, expand=2,
                 window_size=8, shift=False,
                 # HH compensation
                 use_hh_comp=True, hh_lam=0.5, hh_alpha_max=0.5,
                 # w regularizer
                 reg_mode="none", reg_tau=0.0, reg_temp=1.0):
        super().__init__()
        self.c = c
        self.depth = depth
        self.step = step                # kept for API
        self.ws = window_size
        self.shift = shift

        # register Haar kernels as buffers (avoid re-alloc each forward)
        self.register_buffer("haar_h", torch.tensor([1.,  1.]).view(1,1,1,2) / math.sqrt(2), persistent=False)
        self.register_buffer("haar_g", torch.tensor([1., -1.]).view(1,1,1,2) / math.sqrt(2), persistent=False)

        # learnable directional bias
        self.dir_gate = nn.Sequential(
            Conv(self.c, self.c // 4, 1, 1),
                nn.SiLU(),
                Conv(self.c // 4, 4, 1, 1, act=False)  # 4 directions
        )
        # Haar filters as buffers (1x2), transpose() 得 2x1
        h = torch.tensor([1.,  1.], dtype=torch.float32).view(1,1,1,2) / math.sqrt(2)
        g = torch.tensor([1., -1.], dtype=torch.float32).view(1,1,1,2) / math.sqrt(2)
        self.register_buffer("haar_h", h, persistent=False)
        self.register_buffer("haar_g", g, persistent=False)

        # mixers: try mamba, else lightweight 1D mixers
        self.use_mamba = False
        self.mixers = nn.ModuleList()
        if use_mamba:
            try:
                from mamba_module import create_block
                self.use_mamba = True
                for _ in range(depth * 4):
                    self.mixers.append(create_block(c))
            except Exception:
                self.use_mamba = False
        if not self.use_mamba:
            for _ in range(depth * 4):
                self.mixers.append(SeqMixer1D(c, hidden=expand))

        self.agg = GLU2DAggregator(c)

        # HH compensation & regularizer config
        self.use_hh_comp    = bool(use_hh_comp)
        self.hh_lam         = float(hh_lam)
        self.hh_alpha_max   = float(hh_alpha_max)

        self.reg_mode  = str(reg_mode)
        self.reg_tau   = float(reg_tau)
        self.reg_temp  = float(reg_temp)

    # ---------- window helpers ----------
    @staticmethod
    def _window_partition(x, ws, shift=False):
        B, C, H, W = x.shape
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        if shift:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(2, 3))
        nH, nW = H // ws, W // ws
        x = x.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, nH * nW, C, ws, ws)
        meta = (H, W, pad_h, pad_w, nH, nW)
        return x, meta

    @staticmethod
    def _window_reverse(xw, meta, ws, shift=False):
        B, NW, C, ws1, ws2 = xw.shape
        assert ws1 == ws2 == ws
        H, W, pad_h, pad_w, nH, nW = meta
        xw = xw.view(B, nH, nW, C, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = xw.view(B, C, nH * ws, nW * ws)
        if shift:
            x = torch.roll(x, shifts=(ws // 2, ws // 2), dims=(2, 3))
        if pad_h or pad_w:
            x = x[:, :, :H - pad_h, :W - pad_w].contiguous()
        return x

    def _seq4_from_window(self, xw):
        # returns list of four seqs, each (B*NW, L, C), L=ws*ws
        B, NW, C, ws, ws2 = xw.shape
        assert ws == ws2
        s0 = xw.view(B * NW, C, ws * ws).transpose(1, 2)  # →
        s1 = s0.flip(1)                                   # ←
        xw_t = xw.transpose(3, 4).contiguous()
        s2 = xw_t.view(B * NW, C, ws * ws).transpose(1, 2)# ↓
        s3 = s2.flip(1)                                   # ↑
        return [s0, s1, s2, s3]

    def _window_from_seq4(self, seqs, B, NW, C, ws, w=None):
        def to_win(s): return s.transpose(1, 2).contiguous().view(B, NW, C, ws, ws)
        outs = [to_win(seqs[i]) for i in range(4)]
        outs[2] = outs[2].transpose(3, 4).contiguous()
        outs[3] = outs[3].transpose(3, 4).contiguous()
        if w is None:
            y = (outs[0] + outs[1] + outs[2] + outs[3]) / 4.0
        else:
            # stack -> (B,NW,4,C,ws,ws);  w -> (B,NW,4,1,1,1)
            stacked = torch.stack(outs, dim=2)
            y = (w.unsqueeze(3) * stacked).sum(dim=2)
        return y

    # ---------- wavelet helpers ----------
# 放在 LocalSS2DContext 类里，替换 _wavelet_subbands 与 __init__ 中的 h/g 注册
    def _wavelet_subbands(self, x: torch.Tensor):
        B, C, H, W = x.shape
        h  = self.haar_h.to(dtype=x.dtype, device=x.device)   # (1,1,1,2)
        g  = self.haar_g.to(dtype=x.dtype, device=x.device)   # (1,1,1,2)
        hT = h.transpose(2, 3)                                # (1,1,2,1)
        gT = g.transpose(2, 3)                                # (1,1,2,1)

        def sep_conv_same(t: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor):
            t1 = F.pad(t, (1, 0, 0, 0))  # right pad 1 for 1x2
            y  = F.conv2d(t1, kx.expand(C,1,1,2), groups=C, padding=0)
            y1 = F.pad(y, (0, 0, 1, 0))  # bottom pad 1 for 2x1
            y  = F.conv2d(y1, ky.expand(C,1,2,1), groups=C, padding=0)
            return y

        LH = sep_conv_same(x, h,  gT).abs().mean(dim=1, keepdim=True)
        HL = sep_conv_same(x, g,  hT).abs().mean(dim=1, keepdim=True)
        HH = sep_conv_same(x, g,  gT).abs().mean(dim=1, keepdim=True)
        return LH, HL, HH

    def _wavelet_prior_4dir(self, cond: torch.Tensor) -> torch.Tensor:
        LH, HL, _ = self._wavelet_subbands(cond)
        return torch.cat([LH, LH, HL, HL], dim=1)  # [→,←,↓,↑]


    def _w_regularizer(self, w):
        # w: (B,NW,4,1,1) -> (B,NW,4)
        p = w.squeeze(-1).squeeze(-1).clamp_min(1e-8)
        if self.reg_mode == "entropy":
            if self.reg_temp != 1.0:
                q = (p ** (1.0 / self.reg_temp))
                p = q / (q.sum(dim=2, keepdim=True).clamp_min(1e-8))
            H = -(p * (p.log())).sum(dim=2)  # (B,NW)
            return self.reg_tau * H.mean()
        elif self.reg_mode == "sparse":
            l2 = (p ** 2).sum(dim=2)
            return self.reg_tau * (1.0 - l2).mean()
        return p.new_tensor(0.0)

    # ---------- forward ----------
    def forward(self, x, cond=None, return_aux=False):
        B, C, H, W = x.shape
        xw, meta = self._window_partition(x, self.ws, self.shift)
        B1, NW, C1, ws, _ = xw.shape
        assert C1 == self.c

        # 1) four directional sequences
        seqs = self._seq4_from_window(xw)

        # 2) stacked mixers
        for d in range(self.depth):
            for k in range(4):
                seqs[k] = self.mixers[d * 4 + k](seqs[k])

        aux_loss = x.new_tensor(0.0)
        w = None
        if cond is not None:
            # wavelet prior + learnable bias
            prior4 = self._wavelet_prior_4dir(cond)   # (B,4,H,W)
            LH, HL, HH = self._wavelet_subbands(cond) # for uncertainty
            bias4  = self.dir_gate(cond)              # (B,4,H,W)
            gmap   = F.softplus(prior4 + bias4)       # non-negative

            # HH-based uniform compensation
            if self.use_hh_comp:
                denom = (LH + HL + HH).clamp_min(1e-6)
                u = (HH / denom)                      # (B,1,H,W) in [0,1]
                alpha = (self.hh_lam * u).clamp(0.0, self.hh_alpha_max)
                uni = torch.ones_like(gmap) * 0.25
                gmap = (1.0 - alpha) * gmap + alpha * uni

            # window pooling -> w, normalize per window
            gw, _ = self._window_partition(gmap, self.ws, self.shift)  # (B,NW,4,ws,ws)
            w = gw.mean(dim=(3,4), keepdim=True)                       # (B,NW,4,1,1)
            w = w / (w.sum(dim=2, keepdim=True).clamp_min(1e-6))

            # regularizer (training only)
            if self.reg_mode != "none" and self.reg_tau > 0:
                aux_loss = self._w_regularizer(w)

        # 3) aggregate and reverse windows
        yw = self._window_from_seq4(seqs, B1, NW, C1, ws, w=w)
        y  = self._window_reverse(yw, meta, self.ws, self.shift)
        y  = self.agg(y)

        if return_aux:
            return y, {"w_reg_loss": aux_loss}
        return y



class GLU2DAggregator(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(
            Conv(c, c, 3, 1),
            nn.GELU(),
            Conv(c, c, 3, 1, act=False),
        )
        self.fuse = Conv(c, c, 3, 1, act=False)
    def forward(self, x):
        sigma = self.gate(x)
        return self.fuse(sigma * x)
    
class SeqMixer1D(nn.Module):
    def __init__(self, c, hidden=2):
        super().__init__()
        # Treat sequence as (B, L, C): use depthwise separable Conv1d over L.
        self.pw1 = nn.Linear(c, c * hidden)
        self.dw = nn.Conv1d(c * hidden, c * hidden, kernel_size=7, padding=3, groups=c * hidden)
        self.act = nn.SiLU(inplace=True)
        self.pw2 = nn.Linear(c * hidden, c)
        self.norm = nn.LayerNorm(c)
    def forward(self, x):  # x: (B,L,C)
        y = self.pw1(self.norm(x))
        y = self.act(y)
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        y = self.pw2(self.act(y))
        return x + y

# ------------------------------
# JEGO-style scan/merge (single-image adaptation).
def _scan_jego_single(desc, step_size=2):
    # desc: (B,C,H,W)
    desc0 = desc
    desc1 = desc  # single-image adaptation
    desc_2w = torch.cat([desc0, desc1], dim=3)  # (B,C,H,2W)
    desc_2h = torch.cat([desc0, desc1], dim=2)  # (B,C,2H,W)
    B, C, org_h, _ = desc.shape
    _, _, _, org_2w = desc_2w.shape

    # 用 ceil 统一目标长度
    H = math.ceil(org_h / step_size)
    W = math.ceil(org_2w / step_size)
    L = H * W

    def flat(t):  # (B,C,h,w) -> (B,C,h*w)
        return t.contiguous().view(B, C, -1)

    # 四个方向的采样（保持你原来的位移策略）
    right = flat(desc_2w[:, :, ::step_size, ::step_size])                      # (ceil(H/step), ceil(2W/step))
    left  = flat(desc_2h.transpose(2, 3)[:, :, 1::step_size, 1::step_size])    # (ceil(W/step), ceil(2H/step))
    rrev  = flat(desc_2w[:, :, ::step_size, 1::step_size]).flip([2])
    up    = flat(desc_2h.transpose(2, 3)[:, :, ::step_size, 1::step_size]).flip([2])

    def pad_or_crop(x):
        n = x.size(-1)
        if n < L:
            return F.pad(x, (0, L - n))   # 右侧补零
        if n > L:
            return x[:, :, :L]            # 多的裁掉
        return x

    right = pad_or_crop(right)
    left  = pad_or_crop(left)
    rrev  = pad_or_crop(rrev)
    up    = pad_or_crop(up)

    xs = torch.stack([right, left, rrev, up], dim=1).transpose(2, 3)  # (B,4,L,C)
    return xs, org_h, desc.shape[-1]


def _merge_jego_single(ys, ori_h: int, ori_w: int, step_size=2):
    B, _, C, L = ys.shape

    # 1) 和 scan 同一计数：H/W/W2/H2 以及画布尺寸
    H  = math.ceil(ori_h / step_size)
    W  = math.ceil(ori_w / step_size)
    W2 = math.ceil((2 * ori_w) / step_size)
    H2 = math.ceil((2 * ori_h) / step_size)

    new_h,  new_w  = H  * step_size, W  * step_size
    new_2w, new_2h = W2 * step_size, H2 * step_size

    y_2w = torch.zeros((B, C, new_h,  new_2w), device=ys.device, dtype=ys.dtype)
    y_2h = torch.zeros((B, C, new_2h, new_w ), device=ys.device, dtype=ys.dtype)

    # 2) 切片后的“格点数”计算（谁切就按谁来的长度，而不是猜 2*ceil(W/step)）
    def len_from_slice(total, step, start):
        return (total - start + step - 1) // step  # ceil((total - start)/step)

    h_right = len_from_slice(new_h,  step_size, 0)   # == H
    w_right = len_from_slice(new_2w, step_size, 0)   # == W2
    h_rrev  = len_from_slice(new_h,  step_size, 0)   # == H
    w_rrev  = len_from_slice(new_2w, step_size, 1)   # == ceil((new_2w-1)/step)
    h_left  = len_from_slice(new_2h, step_size, 1)   # == ceil((new_2h-1)/step)
    w_left  = len_from_slice(new_w,  step_size, 1)   # == ceil((new_w-1)/step)
    h_up    = len_from_slice(new_2h, step_size, 1)
    w_up    = len_from_slice(new_w,  step_size, 0)   # == W

    # 3) 按目标网格“再对齐”序列长度（pad 或裁剪），然后再 reshape
    def fit1d(x, n):
        l = x.size(-1)
        if l < n:
            x = F.pad(x, (0, n - l))
        elif l > n:
            x = x[..., :n]
        return x

    # right: (H, W2)
    t = fit1d(ys[:, 0], h_right * w_right).reshape(B, C, h_right, w_right)
    y_2w[:, :, ::step_size, ::step_size] = t

    # left: (W, H2) -> transpose 到 (H2, W)
    t = fit1d(ys[:, 1], w_left * h_left).reshape(B, C, w_left, h_left).transpose(2, 3)
    y_2h[:, :, 1::step_size, 1::step_size] = t

    # rrev: (H, W2)
    t = fit1d(ys[:, 2].flip([2]), h_rrev * w_rrev).reshape(B, C, h_rrev, w_rrev)
    y_2w[:, :, ::step_size, 1::step_size] = t

    # up: (W, H2) -> transpose 到 (H2, W)
    t = fit1d(ys[:, 3].flip([2]), w_up * h_up).reshape(B, C, w_up, h_up).transpose(2, 3)
    y_2h[:, :, 1::step_size, ::step_size] = t

    # 4) 再裁回原尺寸，拆半相加
    if (y_2w.shape[-2] != ori_h) or (y_2w.shape[-1] != 2 * ori_w):
        y_2w = y_2w[:, :, :ori_h, :2 * ori_w].contiguous()
    if (y_2h.shape[-2] != 2 * ori_h) or (y_2h.shape[-1] != ori_w):
        y_2h = y_2h[:, :, :2 * ori_h, :ori_w].contiguous()

    desc0_2w, _ = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, _ = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w + desc0_2h
# ------------------------------

class WaveletMixerMultiLevel(nn.Module):
    def __init__(self, c, use_dilated=True, k=5, d=3):
        super().__init__()
        self.dwt  = HaarDWT2D()
        self.idwt = IHaarDWT2D()

        # level-1 subband ops
        self.f_ll1 = Conv(c, c, 1, 1)
        self.f_lh1 = Conv(c, c, 3, 1)
        self.f_hl1 = Conv(c, c, 3, 1)
        self.f_hh1 = Conv(c, c, 3, 1)

        # level-2 (on LL1): adaptive DW conv
        self.use_dilated = use_dilated
        self.k = k
        self.d = d
        self.f_ll2_head = Conv(c, c, 1, 1)
        self.dw_weight = nn.Parameter(torch.empty(c, 1, k, k))
        nn.init.kaiming_uniform_(self.dw_weight, a=math.sqrt(5))
        self.dw_bias = None
        self.f_ll2_tail = Conv(c, c, 1, 1)

        # high-frequency at level-2
        self.f_h2 = Conv(c, c, 3, 1)

    def _depthwise_dynamic(self, x):
        # dynamic dilation to avoid kernel > input
        if not self.use_dilated:
            p = self.k // 2
            return F.conv2d(x, self.dw_weight, self.dw_bias, stride=1,
                            padding=p, dilation=1, groups=x.shape[1])
        H, W = x.shape[-2:]
        d_max_by_size = max(1, (min(H, W) - 1) // (self.k - 1))
        d_dyn = min(self.d, d_max_by_size)
        p = ((self.k - 1) * d_dyn) // 2
        return F.conv2d(x, self.dw_weight, self.dw_bias, stride=1,
                        padding=p, dilation=d_dyn, groups=x.shape[1])

    def forward(self, x):
        # level-1
        LL1, LH1, HL1, HH1 = self.dwt(x)
        LL1 = self.f_ll1(LL1)
        LH1 = self.f_lh1(LH1); HL1 = self.f_hl1(HL1); HH1 = self.f_hh1(HH1)

        # level-2 on LL1
        LL2, LH2, HL2, HH2 = self.dwt(LL1)
        LL2 = self.f_ll2_head(LL2)
        LL2 = self._depthwise_dynamic(LL2)
        LL2 = self.f_ll2_tail(LL2)
        LH2 = self.f_h2(LH2); HL2 = self.f_h2(HL2); HH2 = self.f_h2(HH2)

        # up one level
        LL1_refined = self.idwt(LL2, LH2, HL2, HH2)
        # back to original res
        y = self.idwt(LL1_refined, LH1, HL1, HH1)
        return y


class C3AW_MLM(nn.Module):
    def __init__(self, c1, c2, e=1.0,
                 use_wt: bool = False,
                 wt_levels: int = 1,
                 wt_type: str = 'db1',
                 wt_kernel: int = 5,
                 wt_gamma_init: float = 0.1):
        super().__init__()
        c_ = int(c2 * e)
        self.c_ = c_
        self.use_wt = use_wt

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        if self.use_wt:
            # 仅在 __init__ 中实例化波域算子与门控参数
            self.wt = WTConv2d(c_, c_, kernel_size=wt_kernel, stride=1,
                               wt_levels=wt_levels, wt_type=wt_type)
            self.gamma = nn.Parameter(torch.tensor(float(wt_gamma_init)))
        else:
            self.m = WaveletMixerMultiLevel(c_)

        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)

        # 只做计算，不在 forward 里创建/替换模块
        if self.use_wt:
            # 门控残差：避免波域分支放大噪声
            y_m = x1 + self.gamma * self.wt(x1)
        else:
            y_m = self.m(x1)

        y_s = self.cv2(x)

        # 对齐空间尺寸（优先中心裁剪，其次插值）
        H, W = y_s.shape[-2], y_s.shape[-1]
        if y_m.shape[-2:] != (H, W):
            if y_m.shape[-2] >= H and y_m.shape[-1] >= W:
                dh = (y_m.shape[-2] - H) // 2
                dw = (y_m.shape[-1] - W) // 2
                y_m = y_m[:, :, dh:dh + H, dw:dw + W]
            else:
                y_m = F.interpolate(y_m, size=(H, W), mode='bilinear', align_corners=False)

        return self.cv3(torch.cat((y_m, y_s), dim=1))


def _center_crop_to(x, H, W):
        _, _, h, w = x.shape
        dh = (h - H) // 2
        dw = (w - W) // 2
        return x[:, :, dh:dh+H, dw:dw+W]

class IHaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()
        # 分析滤波器 (保持不变)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        self.register_buffer("ll", ll)
        self.register_buffer("lh", lh)
        self.register_buffer("hl", hl)
        self.register_buffer("hh", hh)
        
        # 新增: Haar 重建滤波器
        recon_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)  # 低通
        recon_h = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)  # 高通
        self.register_buffer("recon_ll", recon_ll.unsqueeze(0).unsqueeze(0))  # [1, 1, 2, 2]
        self.register_buffer("recon_h", recon_h.unsqueeze(0).unsqueeze(0))     # [1, 1, 2, 2]

    def forward(self, LL, LH, HL, HH):
        B, C, Hh, Wh = LL.shape
        # 1) 四个子带对齐到共同的最小高宽
        Hmin = min(LL.shape[-2], LH.shape[-2], HL.shape[-2], HH.shape[-2])
        Wmin = min(LL.shape[-1], LH.shape[-1], HL.shape[-1], HH.shape[-1])
        LL = _center_crop_to(LL, Hmin, Wmin)
        LH = _center_crop_to(LH, Hmin, Wmin)
        HL = _center_crop_to(HL, Hmin, Wmin)
        HH = _center_crop_to(HH, Hmin, Wmin)

        # 2) 拼接 -> (B, 4C, H/2, W/2)
        y = torch.cat([LL, LH, HL, HH], dim=1)

        # 3) 组转置卷积重建，核 (4C,1,2,2)，每通道复用四个 2×2 核
        k = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0)  # (4,2,2)
        k = k.view(4, 1, 2, 2).repeat(C, 1, 1, 1)                     # (4C,1,2,2)
        x = F.conv_transpose2d(y, k, bias=None, stride=2, padding=0, groups=C)
        return x


#class LinearAttention_MSLA(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "LinearAttention: dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        x = (query @ context).reshape(b, h * w, c)

        x = self.proj(x)

        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        return x
    

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x

class MSLA(nn.Module):
    def __init__(self, dim: int = None, num_heads: int = None, *,
                 channels: int = None, heads: int = None, **kwargs):
        """
        统一版 MSLA：
          - 签名支持 dim/num_heads 或 channels/heads
          - forward 同时支持 4D (B,C,H,W) 与 3D (B,N,C)
        """
        super().__init__()
        if dim is None and channels is not None:
            dim = channels
        if num_heads is None and heads is not None:
            num_heads = heads
        assert dim is not None and num_heads is not None, "MSLA 需要 dim 与 num_heads"

        self.dim = int(dim)
        self.num_heads = int(num_heads)

        # 四路深度可分卷积
        self.dw_conv_3x3 = DepthwiseConv(self.dim // 4, 3)
        self.dw_conv_5x5 = DepthwiseConv(self.dim // 4, 5)
        self.dw_conv_7x7 = DepthwiseConv(self.dim // 4, 7)
        self.dw_conv_9x9 = DepthwiseConv(self.dim // 4, 9)

        # 线性注意力在每个 1/4 通道上独立执行
        self.linear_attention = LinearAttention_MSLA(dim=self.dim // 4, num_heads=self.num_heads)

        self.final_conv = nn.Conv2d(self.dim, self.dim, 1)
        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)

    def _forward_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        # x2d: (B,C,H,W)
        B, C, H, W = x2d.shape
        assert C % 4 == 0, f"MSLA 期望通道能被 4 整除，当前 C={C}"

        c4 = C // 4
        x_3x3 = x2d[:, :c4]
        x_5x5 = x2d[:, c4:2*c4]
        x_7x7 = x2d[:, 2*c4:3*c4]
        x_9x9 = x2d[:, 3*c4:]

        att_3x3 = self.linear_attention(self.dw_conv_3x3(x_3x3))
        att_5x5 = self.linear_attention(self.dw_conv_5x5(x_5x5))
        att_7x7 = self.linear_attention(self.dw_conv_7x7(x_7x7))
        att_9x9 = self.linear_attention(self.dw_conv_9x9(x_9x9))

        y2d = torch.cat([
            att_3x3 * self.scale_weights[0],
            att_5x5 * self.scale_weights[1],
            att_7x7 * self.scale_weights[2],
            att_9x9 * self.scale_weights[3],
        ], dim=1)

        return self.final_conv(y2d)  # (B,C,H,W)

    def forward(self, x: torch.Tensor, hw: tuple[int, int] | None = None) -> torch.Tensor:
        """
        - 若 x 是 (B,C,H,W)：直接在 2D 上计算，返回 (B,C,H,W)
        - 若 x 是 (B,N,C)：在 2D 上计算后再还原为 (B,N,C)，允许 N 为非方形面积
        """
        if x.dim() == 4:
            return self._forward_2d(x)  # (B,C,H,W)

        if x.dim() == 3:
            B, N, C = x.shape
            assert C == self.dim, f"MSLA: C 与 dim 不一致，C={C}, dim={self.dim}"
            if hw is None:
                # 自动为非方形序列找一个合理的 (H,W) 因子分解
                H = int(math.sqrt(N))
                while H > 1 and N % H != 0:
                    H -= 1
                W = N // H
            else:
                H, W = hw
                assert H * W == N, f"MSLA: H*W={H*W} 必须等于 N={N}"

            x2d = x.transpose(1, 2).reshape(B, C, H, W)
            y2d = self._forward_2d(x2d)
            return y2d.flatten(2).transpose(1, 2)  # (B,N,C)

        raise ValueError("MSLA.forward 仅支持 (B,C,H,W) 或 (B,N,C)")



    
class DSC3K2_MSLA(C2f):
    """
    DSC3K2_MSLA: 在 C2f/DSC3K2 的隐藏通道上以门控残差方式注入 MSLA。
    - 结构与风格：与 DSC3K2 保持一致（C2f 框架 + 可选 DSBottleneck/DSC3k 堆叠），仅额外增加 MSLA 包装器。
    - 形状契约：输入/输出皆为 (B, C, H, W)，与原链路兼容；可选预处理 stem 支持 stride/dilation/groups。
    - 参数命名：提供 in_channels/out_channels 等直观命名，同时与本文件其他模块风格（c1,c2,n,e,g,shortcut…）保持一致。
    """

    def __init__(
        self,
        # —— 与 block.py 风格一致的主参数（保持与 C2f/DSC3K2 对齐）——
        in_channels: int,               # == c1
        out_channels: int,              # == c2
        n: int = 1,                     # 堆叠深度
        dsc3k: bool = False,            # 内部块类型：False=DSBottleneck, True=DSC3k
        e: float = 0.5,                 # C2f 隐藏通道比例
        g: int = 1,                     # 分组（传给父类；内部 DSBottleneck/DSC3k 本身已定）
        shortcut: bool = True,          # 内部块是否使用残差
        k1: int = 3,                    # DSBottleneck/DSC3k 内部第一层核
        k2: int = 7,                    # DSBottleneck/DSC3k 内部第二层核
        d2: int = 1,                    # DSBottleneck/DSC3k 内部第二层空洞率

        # —— MSLA 融合相关 —— 
        use_msla: bool = True,          # 是否启用 MSLA 注入（默认开启）
        msla_heads: int = 4,            # LinearAttention 头数（传给 MSLA 内部）
        msla_pos: str = "post",         # "post" | "intra" | "none"（默认 post，稳定）
        intra_interval: int = 2,        # "intra" 模式下注入频率（每多少个子块注入一次）
        gamma_init: float = 0.1,        # 门控 γ 的初值（建议小值保证稳定）
        align_by4: bool = True,         # 若 hidden C 不能被4整除，是否做 1x1 临时对齐

        # —— 额外的工程参数（应用户要求暴露，可选使用）——
        hidden_channels: int = None,    # 直接指定隐藏通道数（若给出则覆盖 e* out_channels）
        kernel_size: int = 1,           # 可选的前置 stem 卷积核（默认不改变空间尺寸）
        stride: int = 1,                # 可选的前置 stem 步幅（>1 时会下采样）
        padding: int = None,            # 前置 stem 显式 padding（None 则用 autopad 计算）
        dilation: int = 1,              # 前置 stem 膨胀
        groups: int = 1,                # 前置 stem 分组
        use_bias: bool = False,         # 前置 stem bias
        dropout_rate: float = 0.0,      # MSLA 段 Dropout2d
        norm_type: str = "bn",          # 归一化类型（当前实现仅支持 "bn" 或 "identity"）
    ):
        # —— 初始化 C2f 基类（保持与 DSC3K2 一致）——
        super().__init__(in_channels, out_channels, n, shortcut, g, e)
        # 若指定 hidden_channels，则覆盖父类的 self.c
        if hidden_channels is not None:
            assert hidden_channels > 0, "hidden_channels 必须为正数"
            self.c = hidden_channels
            # 覆盖父类的 cv1/cv2 以适配新的 self.c
            self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, out_channels, 1)

        # —— 可选前置 stem（不改变通道，仅在用户显式设置时启用）——
        need_stem = (kernel_size != 1) or (stride != 1) or (dilation != 1) or (groups != 1)
        if need_stem:
            pad = autopad(kernel_size, padding, dilation)
            stem = [nn.Conv2d(in_channels, in_channels, kernel_size, stride, pad, dilation, groups, bias=use_bias)]
            if norm_type == "bn":
                stem += [nn.BatchNorm2d(in_channels)]
            elif norm_type == "identity":
                pass
            else:
                raise ValueError(f"norm_type 仅支持 'bn' 或 'identity'，收到: {norm_type}")
            stem += [nn.SiLU()]
            self.stem = nn.Sequential(*stem)
        else:
            self.stem = nn.Identity()

        # —— 内部堆叠块：与 DSC3K2 完全一致的分支选择 —— 
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k1=k1, k2=k2, d2=d2)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2)
                for _ in range(n)
            )

        # —— MSLA 包装器：通道×4 对齐 + token 化 + 门控残差 + 可选 Dropout —— 
        self.use_msla = bool(use_msla)
        self.msla_pos = msla_pos
        self.intra_interval = max(1, int(intra_interval))
        self._msla = None
        if self.use_msla and self.msla_pos != "none":
            self._msla = _MSLAWrapperForC2f(
                dim=self.c,
                heads=msla_heads,
                gamma_init=gamma_init,
                align_by4=align_by4,
                dropout_rate=dropout_rate,
            )

        # 形状与设备的基本断言
        assert msla_pos in ("post", "intra", "none"), "msla_pos 只能为 'post' | 'intra' | 'none'"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：与 C2f/DSC3K2 完全一致的主干流程，只在指定位置调用 MSLA 包装器。
        x: (B, C_in, H, W) -> (B, C_out, H', W')，默认 H'=H, W'=W（若 stem 有 stride>1 则会下采样）
        """
        x = self.stem(x)  # 可选的前置处理（不改变通道；可改变空间）

        # C2f 典型流程
        y = list(self.cv1(x).chunk(2, 1))  # y[0], y[1] 各 self.c 通道

        # 堆叠 + 可选 intra 注入
        for i, m in enumerate(self.m):
            y.append(m(y[-1]))
            if self._msla is not None and self.msla_pos == "intra" and ((i + 1) % self.intra_interval == 0):
                y[-1] = self._msla(y[-1])  # 残差形式：内部已做 x + γ·MSLA(x)

        # post 注入（默认最稳）
        if self._msla is not None and self.msla_pos == "post":
            y[-1] = self._msla(y[-1])

        return self.cv2(torch.cat(y, 1))


class _MSLAWrapperForC2f(nn.Module):
    """
    仅用于 C2f 隐藏通道的 MSLA 包装器：
      - 若通道数非4的倍数，使用 1×1 适配到最近的4倍数，再适配回原通道
      - 以 4D (B,C,H,W) 调用 MSLA，避免“必须方形”的约束与不必要的 tokens 往返
      - 残差输出：x + gamma * MSLA(x)
    """
    def __init__(self, dim: int, heads: int, gamma_init: float = 0.1,
                 align_by4: bool = True, dropout_rate: float = 0.0):
        super().__init__()
        self.dim = dim
        self.align = align_by4 and (dim % 4 != 0)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 1e-6 else nn.Identity()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)), requires_grad=True)

        # 1) 对齐到 4 的倍数
        if self.align:
            eff = int((dim + 3) // 4) * 4
            self.in_adapter  = nn.Conv2d(dim, eff, 1, 1, 0, bias=False)
            self.out_adapter = nn.Conv2d(eff, dim, 1, 1, 0, bias=False)
        else:
            eff = dim
            self.in_adapter  = nn.Identity()
            self.out_adapter = nn.Identity()

        # 2) 头数检查：每个分支通道 = eff//4
        self._dim_eff = eff
        assert (eff // 4) % heads == 0, f"LinearAttention 头数不整除：eff={eff}, heads={heads}"

        # 3) 以“位置参数”形式构造，避免关键字不匹配
        self.msla = MSLA(eff, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        u = self.in_adapter(x)                 # (B,C',H,W)
        y = self.msla(u)                       # (B,C',H,W) —— 4D 路径
        y = self.dropout(y)
        y = self.out_adapter(y)                # (B,C,H,W)
        return x + self.gamma * y

# ===== 统一且修复后的 GlobalSparseAttn / SelfAttn / LGLBlock =====
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, padding=1, groups=in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, padding=1, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LocalAgg(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)  # 若工程里有 DropPath
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = x + x * (self.sg(self.pos_embed(x)) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.mlp(self.norm2(x)))) - 0.5)
        return x

class GlobalSparseAttn(nn.Module):
    """
    子采样注意力 + 反卷积还原：
      - 输入 token: N = H*W
      - 若 sr>1：先下采样到 Hs=ceil(H/sr), Ws=ceil(W/sr) 做注意力，再上采样对齐回 (H,W)
      - 返回 token 数恒等于 H*W
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = int(sr_ratio)
        if self.sr > 1:
            # 下采样：ceil_mode=True 以保证整数对齐
            self.sampler = nn.AvgPool2d(kernel_size=self.sr, stride=self.sr, ceil_mode=True)
            # 上采样：kernel=stride=sr，groups=dim 做通道独立上采样
            self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size=self.sr, stride=self.sr, groups=dim, bias=False)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.LocalProp = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: [B, H*W, C]  —— 输入 token
        return: [B, H*W, C]
        """
        B, N, C = x.shape
        assert N == H * W, f"input tokens {N} != H*W {H*W}"

        # 1) 可选下采样到 (Hs, Ws)
        if self.sr > 1:
            feat = x.transpose(1, 2).reshape(B, C, H, W)   # (B,C,H,W)
            feat_ds = self.sampler(feat)                   # (B,C,Hs,Ws) with ceil
            Hs, Ws = feat_ds.shape[-2:]
            x_ds = feat_ds.flatten(2).transpose(1, 2)      # (B, Hs*Ws, C)
        else:
            x_ds = x
            Hs, Ws = H, W

        # 2) 注意力（在 Hs*Ws 个 token 上）
        qkv = self.qkv(x_ds).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (B, num_heads, tokens, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, -1, C)   # (B, Hs*Ws, C)

        # 3) 上采样与对齐（回到 H*W 个 token）
        if self.sr > 1:
            y = y.permute(0, 2, 1).reshape(B, C, Hs, Ws)   # (B,C,Hs,Ws)
            y = self.LocalProp(y)                          # 反卷积上采样，理论 ~ (Hs*sr, Ws*sr)
            if y.shape[-2:] != (H, W):
                # 反卷积可能因 ceil 导致尺寸偏差，强制插值对齐
                y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            y = y.flatten(2).transpose(1, 2)               # (B, H*W, C)
            y = self.norm(y)

        # 4) 投影
        y = self.proj(y)
        y = self.proj_drop(y)

        # 5) （现在才）断言 token 数
        assert y.shape[1] == H * W, f"attn out tokens={y.shape[1]} != H*W={H*W}"
        return y

class SelfAttn(nn.Module):
    """
    注意这里 x 是 [B,C,H,W]，不要将第二维命名为 N。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape                      # 正确的维度次序
        x = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W) # (B,C,H,W)
        return x

class LGLBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.LocalAgg = LocalAgg(dim, mlp_ratio, drop, drop_path, act_layer) if sr_ratio > 1 else nn.Identity()
        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                                 attn_drop, drop_path, act_layer, norm_layer, sr_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x

    
class _DSUnit(nn.Module):
    """
    轻量单元：DW(k1) → PW(1x1) → DW(k2,d2) → PW(1x1)，
    与多数 DSBottleneck 等价（此处固定 in/out=c，stride=1，形状保持）。
    """
    def __init__(self, c: int, k1: int = 3, k2: int = 7, d2: int = 1, shortcut: bool = True):
        super().__init__()
        # 第一组 DSConv（局部聚合 + 通道混合）
        self.ds1 = DSConv(c, c, k=k1, s=1, d=1, bias=False)
        # 第二组 DSConv，可用大核/扩张放大 RF
        self.ds2 = DSConv(c, c, k=k2, s=1, d=d2, bias=False)
        self.add = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ds2(self.ds1(x))
        return x + y if self.add else y
    
class _LGLAdapter(nn.Module):
    """
    将 LGLBlock 以门控残差方式注入：y <- y + gamma * LGL(y)
    - 保持通道数不变：dim=c
    - num_heads: 自动调整为能整除 c 的值，默认为 max(1, c//64)
    - sr_ratio: 建议 >=2 控制注意力 token 成本
    """
    def __init__(
        self,
        c: int,
        num_heads: Optional[int] = None,
        sr_ratio: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = max(1, c // 64)
        # 保障整除
        if c % num_heads != 0:
            # 选择能整除 c 的最接近 num_heads 的因子
            def _nearest_factor(C, h):
                # 找到离 h 最近且能整除 C 的数
                facs = [d for d in range(1, C + 1) if C % d == 0]
                return min(facs, key=lambda x: abs(x - h))
            num_heads = _nearest_factor(c, num_heads)

        self.lgl = LGLBlock(
            dim=c,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=0.0,
            sr_ratio=sr_ratio,
        )
        # 门控因子：0 初始化，训练中逐步放量
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.lgl(x)


class _DSUnitWithLGL(nn.Module):
    """
    将 LGL 以并联残差注入到 _DSUnit 之后： out = DSUnit(x) + gamma * LGL(DSUnit(x))
    - 保持形状与通道不变
    """
    def __init__(self, c: int, k1: int, k2: int, d2: int, shortcut: bool,
                 lgl_heads: Optional[int], lgl_sr: int, lgl_mlp: float, lgl_drop: float, lgl_attn_drop: float):
        super().__init__()
        self.core = _DSUnit(c, k1=k1, k2=k2, d2=d2, shortcut=shortcut)
        self.lgl  = _LGLAdapter(c, num_heads=lgl_heads, sr_ratio=lgl_sr, mlp_ratio=lgl_mlp,
                                drop=lgl_drop, attn_drop=lgl_attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x)
        return self.lgl(y)  # y + gamma·LGL(y)


class DSC3K2_LGL(nn.Module):
    """
    C2f 风格的 DSC3K2 + LGL 并联注入版本：
    - 入口: cv1 将 c1 -> 2c（c=int(c2*e)），split 为 a,b
    - 中间: 对 b 堆叠 n 个 _DSUnitWithLGL(c)，每个保持 c×H×W
    - 末端: concat([a, y1..yn]) -> (2+n)c → cv2 -> c2
    形状契约与原 DSC3K2 一致（可替换使用）。
    """
    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 dsc3k: bool = False,   # 兼容原签名（不使用，仅占位）
                 e: float = 0.5,
                 g: int = 1,            # 兼容原签名（不使用，仅占位）
                 shortcut: bool = True,
                 k1: int = 3,
                 k2: int = 7,
                 d2: int = 1,
                 # LGL 相关新增参数
                 lgl_heads: Optional[int] = None,
                 lgl_sr_ratio: int = 2,
                 lgl_mlp_ratio: float = 4.0,
                 lgl_drop: float = 0.0,
                 lgl_attn_drop: float = 0.0,
                 **kwargs):
        super().__init__()
        assert e > 0, "e must be > 0"
        self.c1, self.c2 = c1, c2
        self.c = int(c2 * e)  # 中间/隐藏通道宽度（与原保持一致）
        # 入口与出口投影（与 C2f/DSC3K2 形状契约一致）
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1)

        # 堆叠 n 个等宽单元，每个单元后并联注入 LGL（门控残差）
        self.m = nn.ModuleList([
            _DSUnitWithLGL(
                c=self.c, k1=k1, k2=k2, d2=d2, shortcut=shortcut,
                lgl_heads=lgl_heads, lgl_sr=lgl_sr_ratio, lgl_mlp=lgl_mlp_ratio,
                lgl_drop=lgl_drop, lgl_attn_drop=lgl_attn_drop
            )
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 C2f 一致：先 1×1 到 2c，再分半为 a,b
        a, b = self.cv1(x).chunk(2, dim=1)  # a,b: [B,c,H,W]
        y = [a, b]
        for block in self.m:
            b = block(b)  # 形状保持 [B,c,H,W]
            y.append(b)
        # 拼接 (2+n)*c → 1×1 投影回 c2
        return self.cv2(torch.cat(y, dim=1))

# 在 block.py 中找到 LinearAttention 的定义，修改其 __init__ 签名：
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=None, qkv_bias=False, proj_bias=True, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, "LinearAttention: dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 如果你的实现是 2D 投影版，建议用原生 Conv2d（上一轮我们已修复过 bias 参数问题）：
        self.qkv  = nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=proj_bias)

    def forward(self, x):               # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        # (B, 3C, H, W) -> (3, B, heads, N, head_dim)
        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]           # (B, heads, N, head_dim)

        # 典型线性注意力：分别在 N 与 head_dim 维度 softmax
        k = F.softmax(k, dim=-1)                   # over head_dim
        q = F.softmax(q, dim=-2)                   # over N

        context = k.transpose(-2, -1) @ v          # (B, heads, head_dim, head_dim)
        y = (q @ context).transpose(2, 3).reshape(B, C, H, W)
        return self.proj(y)

# ==== Add: standard PSABlock (place BEFORE class C2PSA) ====
class PSABlock(nn.Module):
    """
    Position-Sensitive Attention block:
      x = x + Attention(x)
      x = x + FFN(x)
    Args:
      c            : channel dim (must be divisible by num_heads)
      attn_ratio   : key dim ratio wrt head dim (kept for API compatibility)
      num_heads    : heads used in Attention; default ~ c//64 but at least 1
      mlp_ratio    : expansion ratio in the FFN
      qkv_bias     : kept for API compatibility
      proj_bias    : kept for API compatibility
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=None, mlp_ratio=2.0,
                 qkv_bias=True, proj_bias=False, **kwargs):
        super().__init__()
        heads = max(1, (c // 64) if num_heads is None else int(num_heads))
        assert c % heads == 0, f"PSABlock: channels {c} must be divisible by num_heads {heads}"

        # 用你文件里已有的 Attention（保持语义一致）
        self.attn = Attention(c, num_heads=heads, attn_ratio=attn_ratio)

        hidden = int(c * mlp_ratio)
        # 与文件风格一致，用 1x1 Conv 做 FFN
        self.ffn = nn.Sequential(
            Conv(c, hidden, k=1, s=1, act=True),
            Conv(hidden, c, k=1, s=1, act=False),
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x
# ==== End add ====


class PSABlock_LinearAttention(nn.Module):
    """
    PSA Block with Linear Attention:
      x = x + LinearAttention(x)
      x = x + FFN(x)
    """
    def __init__(
        self,
        dim: int,
        attn_ratio: float = 0.5,
        num_heads: int = None,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        proj_bias: bool = False,
        fmap: str = "elu",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = LinearAttention(
            dim=dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            fmap=fmap,
            eps=eps,
        )
        hidden = int(dim * mlp_ratio)
        # 用 Conv 做 1x1 MLP（与原 PSABlock 风格一致）
        self.ffn = nn.Sequential(
            Conv(dim, hidden, k=1, s=1, act=True),
            Conv(hidden, dim, k=1, s=1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C2PSA_LinearAttention(nn.Module):
    """
    CSP-2 分流 + 线性注意力堆叠 + 回并
    与原 C2PSA 的接口一致：
      - __init__(c1, c2, n=1, e=0.5, ...)
      - forward(x): (B,C,H,W) -> (B,C,H,W)
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        attn_ratio: float = 0.5,
        num_heads: int = None,
        mlp_ratio: float = 2.0,
        fmap: str = "elu",
    ):
        super().__init__()
        assert c1 == c2, "C2PSA_LinearAttention 要求 c1 == c2（保持通道不变）"
        self.c = int(c1 * e)
        # 安全的 head 设定（在 PSABlock_LinearAttention 内部也会再次检查）
        heads = max(1, (self.c // 64) if num_heads is None else num_heads)
        assert self.c % heads == 0, f"分支通道 {self.c} 必须能够被 num_heads {heads} 整除"

        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.m = nn.Sequential(*[
            PSABlock_LinearAttention(
                dim=self.c,
                attn_ratio=attn_ratio,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                fmap=fmap
            ) for _ in range(n)
        ])
        self.cv2 = Conv(2 * self.c, c1, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) -> (B,C,H,W)
        """
        y = self.cv1(x)                       # (B, 2c, H, W)
        a, b = torch.split(y, (self.c, self.c), dim=1)  # a:直连，b:注意力支路
        b = self.m(b)                         # (B, c, H, W)
        out = torch.cat((a, b), dim=1)        # (B, 2c, H, W)
        return self.cv2(out)

class C3k2_TWavelet(nn.Module):
    """
    C3k2 + Wavelet 子带增强（可直接替换 C3k2）
    签名保持与 C3k2 一致: (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)
    设计要点：
      - 在 C2f 框架内，对进入堆叠的分支 b 做一次 Haar DWT 子带分解与轻处理，再融合回同形状的张量，继续原有堆叠。
      - 不更改 (2+n)*c 的拼接逻辑与 cv2 投影，保证对外接口与形状完全兼容。
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()

        self.c = max(1, int(c2 * e))               # 隐通道，防 0
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 堆叠分支（保持与 C3k2 相同的可选 C3k/Bottleneck）
        block_ctor = (lambda cin, cout, sc=shortcut, gg=g: C3k(cin, cout, 1, sc, gg)) if c3k \
                     else (lambda cin, cout, sc=shortcut, gg=g: Bottleneck(cin, cout, sc, gg, k=((3, 3), (3, 3)), e=1.0))
        self.m = nn.ModuleList(block_ctor(self.c, self.c) for _ in range(n))

        # Wavelet 子带增强（用你文件中的 HaarDWT2D）
        self.dwt = HaarDWT2D()
        self.f_ll = Conv(self.c, self.c // 2, 1, 1)   # 低频偏结构化：1x1
        self.f_h  = Conv(self.c, self.c // 2, 3, 1)   # 高频（LH/HL/HH）共享 3x3
        self.fuse = Conv(3 * self.c, self.c, 1, 1)    # [b, 4*sub] -> c
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.2, 0.2, 0.1], dtype=torch.float32))  # [LL,LH,HL,HH]
        self.gamma = nn.Parameter(torch.tensor(0.0))   # 小残差缩放，稳定训练

    @staticmethod
    def _upsample(x: torch.Tensor, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def _safe_dwt(self, x: torch.Tensor):
        # 奇偶尺寸安全处理：reflect pad -> DWT -> 裁回
        B, C, H, W = x.shape
        pad_h, pad_w = H & 1, W & 1
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        LL, LH, HL, HH = self.dwt(x)  # (B,C,H/2,W/2) * 4
        if pad_h or pad_w:
            LL = LL[..., :H//2, :W//2]
            LH = LH[..., :H//2, :W//2]
            HL = HL[..., :H//2, :W//2]
            HH = HH[..., :H//2, :W//2]
        return LL, LH, HL, HH

    def _wave_enhance(self, b: torch.Tensor) -> torch.Tensor:
        # 仅增强堆叠分支 b： (B,c,H,W) -> (B,c,H,W)
        B, C, H, W = b.shape
        LL, LH, HL, HH = self._safe_dwt(b)

        # 子带轻处理（在 H/2,W/2 上，算力开销小）
        LLp = self.f_ll(LL)
        LHp = self.f_h(LH)
        HLp = self.f_h(HL)
        HHp = self.f_h(HH)

        # 非负归一化权重
        w = F.softplus(self.alpha)
        w = w / (w.sum() + 1e-6)

        # 上采样回原尺寸并按权重缩放
        size = (H, W)
        LLu = self._upsample(LLp, size) * w[0]
        LHu = self._upsample(LHp, size) * w[1]
        HLu = self._upsample(HLp, size) * w[2]
        HHu = self._upsample(HHp, size) * w[3]

        # 拼接后用 1x1 融合回 c，再以一个小系数注入残差
        y = torch.cat([b, LLu, LHu, HLu, HHu], dim=1)  # (B, 3c, H, W)
        y = self.fuse(y)
        return b + self.gamma.tanh() * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 C2f 保持一致的前向流程
        y = list(self.cv1(x).chunk(2, 1))      # [a,b], 各 (B,c,H,W)
        y[1] = self._wave_enhance(y[1])        # 只增强堆叠链路
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ====== Wavelet-Enhanced C2f-family modules (pywt enabled by default) ======
class _PywtDWT2D(nn.Module):
    """
    2D DWT（分解）模块：使用 pywt 提供的小波基系数，转为可微分的 depthwise conv2d 计算。
    - 支持任意 pywt.Wavelet 名称（如 'haar', 'db2', 'sym4', 'coif1', 'bior2.2', 'rbio2.2', 'dmey' 等）
    - 仅执行一级分解，输出 4 个子带 (LL, LH, HL, HH)
    - 采用对称边界（symmetric）近似：使用 reflect padding 实现
    - 保持梯度流：所有计算在 PyTorch 中以卷积实现，不调用 numpy 进行正向
    """

    def __init__(self, wave: str = "haar", mode: str = "symmetric"):
        super().__init__()
        self.wave_name = wave
        self.mode = mode  # 目前用于指示 padding 策略，'symmetric' -> reflect

        # 从 pywt 读取分解滤波器系数（1D）
        w = pywt.Wavelet(self.wave_name)
        h0 = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)  # low-pass (reversed for conv)
        h1 = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)  # high-pass

        # 生成 2D 滤波器（外积）
        # LL = h0^T ⊗ h0, LH = h0 ⊗ h1, HL = h1 ⊗ h0, HH = h1 ⊗ h1
        kLL = torch.einsum('i,j->ij', h0, h0).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        kLH = torch.einsum('i,j->ij', h0, h1).unsqueeze(0).unsqueeze(0)
        kHL = torch.einsum('i,j->ij', h1, h0).unsqueeze(0).unsqueeze(0)
        kHH = torch.einsum('i,j->ij', h1, h1).unsqueeze(0).unsqueeze(0)

        # 按 (4,1,k,k) 堆叠，作为基础权重；前向时会按通道 C 重复并进行 depthwise 分组卷积
        weight = torch.cat([kLL, kLH, kHL, kHH], dim=0)  # (4,1,k,k)
        self.register_buffer("weight", weight, persistent=False)

        # 记录核大小，用于 padding 计算
        self.k = int(weight.shape[-1])
        # 对称填充长度（反射）：对奇/偶核分别给出常用近似
        # - 奇核：pad = k//2
        # - 偶核：pad = k//2 - 1    （pywt 的 'symmetric' 与 conv 的 reflect 并非完全等价，此为经验近似）
        self.pad_each_side = self.k // 2 if (self.k % 2 == 1) else max(self.k // 2 - 1, 0)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        return: LL, LH, HL, HH (四个张量，形状为 (B, C, H//2, W//2) 或 ceil 降采样)
        """
        B, C, H, W = x.shape
        w = self.weight.to(dtype=x.dtype, device=x.device)  # (4,1,k,k)

        # 反射 padding 以近似 'symmetric' 边界条件，并使 stride=2 的 conv 输出更稳定
        pad = self.pad_each_side
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        # 将 (4,1,k,k) 扩展为 (4C,1,k,k)，实现跨通道 depthwise 组卷积；groups=C
        w_rep = w.repeat(C, 1, 1, 1)  # (4C,1,k,k)
        y = F.conv2d(x, w_rep, bias=None, stride=2, padding=0, groups=C)  # (B, 4C, H', W')

        # 重排输出到 (B, C, 4, H', W')，再拆分为 4 个子带
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        LL = y[:, :, 0, ...]
        LH = y[:, :, 1, ...]
        HL = y[:, :, 2, ...]
        HH = y[:, :, 3, ...]
        return LL, LH, HL, HH


class _WaveletEnhancer(nn.Module):
    """
    子模块：对 (B, c, H, W) 进行 DWT(来自 pywt 的小波基系数) -> 子带轻处理 -> 上采样融合 -> 残差注入
    - 默认启用 pywt：wave='haar'，mode='symmetric'
    - 仅增强 C2f 的堆叠链路分支 b，降低算力与显存压力
    - 参数:
        c         : 通道数
        use_ds    : 高频子带是否使用 DSConv（True 更轻量；False 使用 Conv）
        alpha0    : 初始子带权重（顺序: [LL, LH, HL, HH]），经 softplus 再归一化
        wave      : 小波名称（pywt.Wavelet），如 'haar'/'db2'/'sym4'/'coif1'/'bior2.2'/'rbio2.2'/'dmey' 等
        mode      : 边界模式占位，目前使用 'symmetric' -> reflect padding
    """
    def __init__(
        self,
        c: int,
        use_ds: bool = False,
        alpha0=(0.5, 0.2, 0.2, 0.1),
        wave: str = "haar",
        mode: str = "symmetric",
    ):
        super().__init__()
        self.c = c
        self.dwt = _PywtDWT2D(wave=wave, mode=mode)

        # 低频采用 1x1；高频共享 3x3/DSConv
        from .conv import Conv, DSConv  # 放在此处以避免循环导入（若当前文件为 block.py 可移到顶部）
        self.f_ll = Conv(c, c // 2, k=1, s=1)
        self.f_h  = (DSConv if use_ds else Conv)(c, c // 2, k=3, s=1)

        # 融合回 c 通道（b + 4*up(sampled) -> 与 b 拼接后 1x1 融合）
        self.fuse = Conv(3 * c, c, k=1, s=1)

        # 可学习子带权重与注入强度
        self.alpha = nn.Parameter(torch.tensor(alpha0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _upsample(x: torch.Tensor, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = b.shape
        # DWT（pywt 小波系数 -> depthwise conv 实现，可微）
        LL, LH, HL, HH = self.dwt(b)

        # 子带轻处理（在 H/2, W/2 上完成）
        LLp = self.f_ll(LL)
        LHp = self.f_h(LH)
        HLp = self.f_h(HL)
        HHp = self.f_h(HH)

        # 非负归一化权重
        w = F.softplus(self.alpha)
        w = w / (w.sum() + 1e-6)

        # 上采样回原尺度并加权
        size = (H, W)
        LLu = self._upsample(LLp, size) * w[0]
        LHu = self._upsample(LHp, size) * w[1]
        HLu = self._upsample(HLp, size) * w[2]
        HHu = self._upsample(HHp, size) * w[3]

        # 拼接并用 1x1 融合回 c，最后以 gamma 注入残差
        y = torch.cat([b, LLu, LHu, HLu, HHu], dim=1)  # (B, 3c, H, W)
        y = self.fuse(y)
        return b + self.gamma.tanh() * y


class C3k2_Wavelet(nn.Module):
    """
    C2f 风格的 C3k2 + Wavelet 模块（子模块）
    - 签名保持与 C3k2 一致: (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)
    - 前向流程与 C2f 一致，但仅对堆叠链路 b 注入 Wavelet 增强
    - 新增（向后兼容）：可通过 kwargs 传入 wave='db2'/mode='symmetric'/use_ds=True 等；默认 wave='haar'
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, **kwargs):
        super().__init__()
        from .conv import Conv  # 避免循环导入（若本文件为 block.py 可移到顶部）
        from .block import Bottleneck, C3k  # 若当前为 block.py，可直接引用

        use_ds = bool(kwargs.get("use_ds", False))
        wave   = kwargs.get("wave", "haar")
        mode   = kwargs.get("mode", "symmetric")

        self.c = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        # 选择内部块：与 C3k2 保持一致
        block_ctor = (lambda cin, cout, sc=shortcut, gg=g: C3k(cin, cout, 2, sc, gg)) if c3k \
                     else (lambda cin, cout, sc=shortcut, gg=g: Bottleneck(cin, cout, sc, gg, k=((3, 3), (3, 3)), e=1.0))
        self.m = nn.ModuleList(block_ctor(self.c, self.c) for _ in range(n))

        # Wavelet 增强器（默认启用 pywt，小波基可配）
        self.wave = _WaveletEnhancer(self.c, use_ds=use_ds, wave=wave, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))      # [a, b], 各 (B, c, H, W)
        y[1] = self.wave(y[1])                 # 仅增强堆叠链路 b
        for m in self.m:
            y.append(m(y[-1]))                 # 堆叠 n 次
        return self.cv2(torch.cat(y, 1))


class DSC3K2_Wavelet(nn.Module):
    """
    与 DSC3K2 同签名的 Wavelet 增强版本（完全对齐外部接口）
    - 签名: (c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1, **kwargs)
    - 默认不更改 YAML/调用方式；如需切换到 Wavelet 版本，直接将模块名替换为 DSC3K2_Wavelet 即可
    - Wavelet 注入点：仅对 b 分支
    - 新增（向后兼容）：kwargs 支持 wave='db2'/mode='symmetric'/use_ds=True 等；默认 wave='haar'
    """
    def __init__(
        self,
        c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True,
        k1=3, k2=7, d2=1, **kwargs
    ):
        super().__init__()
        
        use_ds = bool(kwargs.get("use_ds", False))
        wave   = kwargs.get("wave", "haar")
        mode   = kwargs.get("mode", "symmetric")

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        # 内部块：与 DSC3K2 保持一致（dsc3k=True -> 使用 DSC3k；否则使用 DSBottleneck）
        if dsc3k:
            self.m = nn.ModuleList(DSC3k(self.c, self.c, n=2, shortcut=shortcut, g=g) for _ in range(n))
        else:
            self.m = nn.ModuleList(
                DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n)
            )

        # Wavelet 增强器（默认启用 pywt，小波基可配）
        self.wave = _WaveletEnhancer(self.c, use_ds=use_ds, wave=wave, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))  # [a, b]
        y[1] = self.wave(y[1])             # 仅增强 b
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

