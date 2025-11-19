# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Activation modules."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))

# ===== 核心的函数式实现 =====
@torch.jit.ignore
def _stable_tanh_exp(x: torch.Tensor, cutoff: float = 20.0) -> torch.Tensor:
    """
    计算 tanh(exp(x)) 的数值稳定版本：
    - 对于 x >> 0，tanh(exp(x)) ≈ 1，因此直接返回 1，避免 exp 溢出
    - 对于其余区域，安全地计算 tanh(exp(clamp(x, max=cutoff)))
    """
    # 大正值近似：tanh(exp(x)) ≈ 1
    large_pos = x > cutoff
    if large_pos.any():
        # 其余位置按正常公式计算，但把 x 限制到 cutoff 以内避免 exp 溢出
        e = torch.exp(torch.clamp(x, max=cutoff).to(torch.float32)).to(x.dtype)
        t = torch.tanh(e)
        t = torch.where(x > cutoff, torch.ones_like(x), t)
    else:
        e = torch.exp(torch.clamp(x, max=cutoff))
        out = torch.tanh(e)
    return out

def telu(x: torch.Tensor, stable: bool = True, cutoff: float = 20.0) -> torch.Tensor:
    """
    TeLU 激活：y = x * tanh(exp(x))
    参考论文给出的定义与导数形式。:contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

    参数
    ----
    x : Tensor
    stable : bool
        是否启用数值稳定版本（推荐）。对大正数使用近似 y≈x（因 tanh(exp(x))≈1），
        对极负数采用安全的 exp 限幅计算。
    cutoff : float
        稳定计算的正向截断阈值（float32 下 20 已足够让 tanh(exp(x))≈1）。

    返回
    ----
    y : Tensor
    """
    if stable:
        t = _stable_tanh_exp(x, cutoff)
        return x * t
    else:
        # 直接的数学定义（可能在大正数溢出到 inf 但仍可用）
        return x * torch.tanh(torch.exp(x))

# ===== 自定义 autograd（可选）：在稳定近似下提供解析反向，避免重复图计算 =====
class _TeLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cutoff: float):
        # 前向：与 telu(stable=True) 等价
        large_pos = x > cutoff
        xc = torch.clamp(x, max=cutoff)
        e = torch.exp(xc.to(torch.float32)).to(x.dtype)
        t = torch.tanh(e)
        t = torch.where(large_pos, torch.ones_like(x), t)  # tanh(exp(x))≈1
        y = x * t
        # 保存用于反向的中间量：
        # 论文中的一阶导：tanh(e^x) + x * e^x * (1 - tanh^2(e^x))。:contentReference[oaicite:4]{index=4}
        ctx.save_for_backward(x, t, e, large_pos)
        ctx.cutoff = cutoff
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, t, e, large_pos = ctx.saved_tensors
        # 对 large_pos：t≈1, 导数≈1（因为 y≈x）
        # 其余区域使用解析导数：t + x * e * (1 - t^2)
        one = torch.ones_like(x)
        sech2 = (one - t * t)  # 1 - tanh^2
        grad_local = torch.where(
            large_pos,
            one,                       # dy/dx ≈ 1
            t + x * e * sech2          # 论文给出的精确导数
        )
        return grad_out * grad_local, None

# ===== nn.Module 封装 =====
class TeLU(nn.Module):
    """
    PyTorch 模块版的 TeLU。
    - 默认启用数值稳定路径（并带有自定义反向）
    - 如果你更偏好完全由 autograd 推导，可设 use_custom_backward=False
    """
    def __init__(self, stable: bool = True, cutoff: float = 20.0, use_custom_backward: bool = True):
        super().__init__()
        self.stable = stable
        self.cutoff = float(cutoff)
        self.use_custom_backward = bool(use_custom_backward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stable:
            if self.use_custom_backward and x.requires_grad:
                return _TeLUFunc.apply(x, self.cutoff)
            else:
                return telu(x, stable=True, cutoff=self.cutoff)
        else:
            return telu(x, stable=False)

# ===== 便捷测试与示例 =====
if __name__ == "__main__":
    x = torch.linspace(-10, 10, 5, requires_grad=True)
    act = TeLU()  # 默认稳定 + 自定义反向
    y = act(x)
    y.sum().backward()
    print("x:", x.detach())
    print("y:", y.detach())
    print("dy/dx:", x.grad)

    # 与纯函数式调用
    z = telu(x.detach(), stable=True)
    print("telu(x):", z)

    # 集成到模型
    m = nn.Sequential(
        nn.Linear(16, 32),
        TeLU(),          # 直接替换 ReLU/SiLU
        nn.Linear(32, 10)
    )
    dummy = torch.randn(4, 16)
    out = m(dummy)
    print(out.shape)

    # TorchScript（当 use_custom_backward=False 时更易脚本化）
    m_script = nn.Sequential(
        nn.Linear(16, 32),
        TeLU(stable=True, use_custom_backward=False),  # 建议这样以便 torchscript
        nn.Linear(32, 10)
    )
    scripted = torch.jit.script(m_script)
    scripted_out = scripted(dummy)
    print("scripted ok:", scripted_out.shape)