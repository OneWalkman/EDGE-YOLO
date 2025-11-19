#!/usr/bin/env python
# Test.py  --  彻底在 GPU 上初始化 YOLOv13-LSNet

import os, sys, torch

# ① 先指定默认设备和 Ultralytics 设备，再导入任何 ultralytics 相关包
torch.set_default_device("cuda")                 # PyTorch ≥2.1 推荐做法
os.environ["ULTRALYTICS_DEVICE"] = "cuda:0"      # Ultralytics 自己也会读取

# ② 把项目根目录加入 PYTHONPATH，然后再 import
sys.path.insert(0, "/root/workspace/Yolov13-Lsnet")
from ultralytics import YOLO

# ③ 构建模型（此时 Ultralytics 整个 build 流程已经全在 GPU）
model = YOLO("ultralytics/cfg/models/v13/yolov13-lsnet.yaml")

# ④ 如果你后面还想手动迁移其它 tensor，可显式写一行保险：
# model.model = model.model.cuda()   # 理论上现在已经不需要

model.info()
print("模型全部在设备：", next(model.model.parameters()).device)  # 应当打印 cuda:0
