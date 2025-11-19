# debug_attn.py
import torch
from ultralytics.nn.modules import ska          # ① 先导入 ska

# ---------- 给所有 SKA 派生类打补丁 ----------
def dbg_ska(self, x, w):
    print(f"[SKA dbg]  x.device={x.device} | w.device={w.device}")
    return x    # 直接返回，跳过 CUDA kernel

# ↑ 必须在 import LSNet 前完成
for name, obj in ska.__dict__.items():
    if isinstance(obj, type) and obj.__name__.startswith("SKA"):
        obj.forward = dbg_ska

# ---------- 现在才导入 LSNet ----------
from ultralytics.nn.modules.lsnet import LSNet, Attention

# 可选：给 Attention 也加个打印
def dbg_attn(self, x):
    print(f"[ATTN dbg] x.device={x.device}")
    return x
Attention.forward = dbg_attn
# -------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net    = LSNet().to(device).eval()
dummy  = torch.randn(1, 3, 64, 64, device=device)

with torch.no_grad():
    net(dummy)
