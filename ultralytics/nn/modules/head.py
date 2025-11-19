# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "GF2Detect", "E2EDetect","GFLHeadv2_uniH","GFLHeadv2_E2E",

# head.py 里新增一个小模块
class DGQP(nn.Module):
    def __init__(self, k=4, p=64):
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(4*(k+1), p)  # 4 个边，每边 top-k + mean
        self.fc2 = nn.Linear(p, 1)

    def forward(self, dist_softmax):  
        # dist_softmax: (B, A, 4, reg_max) 概率分布
        topk_vals, _ = dist_softmax.topk(self.k, dim=-1)          # (B,A,4,k)
        mean_vals = topk_vals.mean(dim=-1, keepdim=True)          # (B,A,4,1)
        stat = torch.cat([topk_vals, mean_vals], dim=-1)          # (B,A,4,k+1)
        stat = stat.reshape(stat.size(0), stat.size(1), -1)        # (B,A,4*(k+1))
        x = F.relu(self.fc1(stat))
        I = torch.sigmoid(self.fc2(x)).squeeze(-1)                # (B,A)
        return I

class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
        
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

# 假设与 Detect 同一文件或已正确 import Detect / Conv / DFL / make_anchors / dist2bbox 等
# import copy, torch, torch.nn as nn

class GF2Detect(Detect):
    """
    GFocalV2-style head built on top of Detect:
      - 不重复 Detect 中已有参数/结构（回归/分类塔、DFL、anchors/strides 等均复用）
      - 保持与 Detect 完全一致的 I/O 接口与张量形状
      - 在 DFL 积分之前，对回归分布做 LQE/DGQP 统计，得到 quality 分数；
        推理时将 quality 乘到分类概率上，提高分数与定位质量的一致性
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)  # 复用 Detect 的一切既有定义

        # ===== 仅新增与 LQE/DGQP 相关的超参与小头（每层一套） =====
        self.reg_topk = 4                 # 从每条边的分布中取 top-k 概率
        self.add_mean = True              # 是否拼接均值
        self.reg_channels = 64            # 质量小头的中间通道
        self.apply_quality_in_inference = True  # 推理时是否将 quality 乘到分类概率

        in_stat = 4 * (self.reg_topk + (1 if self.add_mean else 0))  # 4条边 × (topk [+ mean])
        self.reg_conf = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_stat, self.reg_channels, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.reg_channels, 1, 1, bias=True),
                nn.Sigmoid(),
            ) for _ in ch
        )
        if self.end2end:
            self.one2one_reg_conf = copy.deepcopy(self.reg_conf)

        # 供推理阶段临时缓存（不改变对外接口）
        self._qualities = None

    # ---------- LQE/DGQP: 从回归分布 logits 计算质量图 ----------
    def _compute_quality_from_logits(self, box_logits: torch.Tensor, head_idx: int, branch: str = "one2many"):
        """
        box_logits: 形状 (B, 4*reg_max, H, W)，为 DFL 积分之前的回归分布 logits
        返回 q: 形状 (B, 1, H, W)，范围 [0,1]
        """
        B, _, H, W = box_logits.shape
        # 在 bin 维（K=reg_max）做 softmax 得概率分布
        prob = box_logits.view(B, 4, self.reg_max, H, W).softmax(dim=2)  # (B,4,K,H,W)
        k = min(self.reg_topk, self.reg_max)
        topk = torch.topk(prob, k=k, dim=2).values                       # (B,4,k,H,W)
        stat_parts = [topk]
        if self.add_mean:
            stat_parts.append(prob.mean(dim=2, keepdim=True))            # (B,4,1,H,W)
        stat = torch.cat(stat_parts, dim=2).view(B, -1, H, W)            # (B, 4*(k+mean), H, W)

        head = self.one2one_reg_conf if (branch == "one2one" and hasattr(self, "one2one_reg_conf")) else self.reg_conf
        return head[head_idx](stat)                                      # (B,1,H,W)

    @staticmethod
    def _cat_quality(qualities, B):
        """将每层的 (B,1,H,W) quality 拼接成 (B,1,A)，与 Detect._inference 的 A 对齐"""
        return torch.cat([q.view(B, 1, -1) for q in qualities], dim=2) if qualities else None

    # ---------------- 覆写 forward：保持 I/O 与 Detect 一致 ----------------
    def forward(self, x):
        if self.end2end:
            return self.forward_end2end(x)

        # 计算每层回归/分类输出，并在内部得到 quality（不改变返回接口）
        qualities = []
        for i in range(self.nl):
            box_i = self.cv2[i](x[i])                  # (B, 4*reg_max, H, W)
            cls_i = self.cv3[i](x[i])                  # (B, nc,        H, W)
            qualities.append(self._compute_quality_from_logits(box_i, i, "one2many"))
            x[i] = torch.cat((box_i, cls_i), 1)

        if self.training:
            self._qualities = qualities  # ✅ 训练态缓存，供 loss 端读取
            return x

        # 推理期：缓存 quality，并在解码时对分类概率做质量调制
        self._qualities = qualities
        y = self._inference_with_quality(x, qualities)
        return y if self.export else (y, x)

    # ---------------- 覆写 E2E 前向：接口与 Detect 完全一致 ----------------
    def forward_end2end(self, x):
        x_detach = [xi.detach() for xi in x]

        # one-to-one（冻结特征）分支
        one2one_feats, q_one2one = [], []
        for i in range(self.nl):
            b_i = self.one2one_cv2[i](x_detach[i])
            c_i = self.one2one_cv3[i](x_detach[i])
            q_one2one.append(self._compute_quality_from_logits(b_i, i, "one2one"))
            one2one_feats.append(torch.cat((b_i, c_i), 1))

        # one-to-many（可学习）分支
        one2many_feats = []
        for i in range(self.nl):
            b_i = self.cv2[i](x[i])
            c_i = self.cv3[i](x[i])
            one2many_feats.append(torch.cat((b_i, c_i), 1))

        if self.training:
            # 训练期：保持与 Detect 一致（不返回 qualities）
            return {"one2many": one2many_feats, "one2one": one2one_feats}

        # 推理：默认使用 one-to-one 分支，且做质量调制
        y = self._inference_with_quality(one2one_feats, q_one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": one2many_feats, "one2one": one2one_feats})

    # ---------------- 仅在本类中使用：与 Detect._inference 等价但加入 quality ----------------
    def _inference_with_quality(self, x, qualities):
        """Decode boxes + apply quality on class probabilities. 输出形状与 Detect._inference 完全一致。"""
        shape = x[0].shape  # BCHW
        B = shape[0]
        x_cat = torch.cat([xi.view(B, self.no, -1) for xi in x], 2)

        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 拆分回归/分类
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # 拼接 quality 到 (B,1,A)
        q_cat = self._cat_quality(qualities, B)

        # 导出分支与默认分支：与 Detect 一致，仅在分类概率处乘以 q_cat
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
            cls_prob = cls.sigmoid()
            if self.apply_quality_in_inference and q_cat is not None:
                cls_prob = cls_prob * q_cat.clamp(1e-6, 1 - 1e-6)
            return torch.cat((dbox, cls_prob), 1)

        if self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            cls_prob = cls.sigmoid()
            if self.apply_quality_in_inference and q_cat is not None:
                cls_prob = cls_prob * q_cat.clamp(1e-6, 1 - 1e-6)
            return dbox.transpose(1, 2), cls_prob.permute(0, 2, 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        cls_prob = cls.sigmoid()
        if self.apply_quality_in_inference and q_cat is not None:
            cls_prob = cls_prob * q_cat.clamp(1e-6, 1 - 1e-6)
        return torch.cat((dbox, cls_prob), 1)

class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

class E2EDetect(GF2Detect):
    """
    v10 Detection head (E2E) — 仅覆写分类头为轻量DW结构。
    其余功能（DFL、LQE/DGQP质量估计、E2E训练/推理）全部复用父类 Detect 的实现。
    """
    end2end = True  # 继承时即启用 E2E 分支；父类 __init__ 会据此构建 one2one 分支与质量小头

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)  # 构建回归/分类头(默认版)、DFL、LQE/DGQP 小头、E2E 分支等

        # -------- 仅替换“分类头”为 v10 的轻量结构 --------
        # 说明：
        # - 保留父类 self.cv2（回归分支，输出4*reg_max的分布logits，供DFL与LQE使用）
        # - 将 self.cv3 改为 v10 的DW可分离 + 1x1结构，最后输出 nc 通道
        # - 替换后需同步刷新 E2E 的 one2one_cv3，保持两路结构一致
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x),  # DWConv
                              Conv(x, c3, 1)),     # PW 1x1
                nn.Sequential(Conv(c3, c3, 3, g=c3),
                              Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            ) for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)  # 同步E2E分支的分类头


class GFLHeadv2_uniH(GF2Detect):
    def __init__(self, nc=80, ch=(), reg_topk=4, add_mean=True, reg_channels=64,
                 use_dat=False, use_cit=False, use_poscnn=False):
        super().__init__(nc, ch)
        # 可选：你的特征处理模块（占位/替换为真实实现）
        block = nn.Identity
        self.stem = nn.ModuleList(block() for _ in ch)  # 可替换为 DCNv2 stem
        self.dat  = nn.ModuleList(block() for _ in ch) if use_dat else None
        self.pos_cls = nn.ModuleList(block() for _ in ch) if use_poscnn else None
        self.pos_reg = nn.ModuleList(block() for _ in ch) if use_poscnn else None
        self.cit_cls = nn.ModuleList(block() for _ in ch) if use_cit else None
        self.cit_reg = nn.ModuleList(block() for _ in ch) if use_cit else None

        # 覆盖/对齐 GF2Detect 的质量头超参（与 mmdet 版一致）
        self.reg_topk = reg_topk
        self.add_mean = add_mean
        self.reg_channels = reg_channels

        # 质量小头：每层一套（输入通道 = 4*(topk + mean_flag)）
        in_stat = 4 * (self.reg_topk + (1 if self.add_mean else 0))
        self.reg_conf = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_stat, self.reg_channels, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.reg_channels, 1, 1, bias=True),
                nn.Sigmoid(),
            ) for _ in ch
        )
        
    def _dgqp_quality(self, reg_logits: torch.Tensor, i: int) -> torch.Tensor:
        """
        reg_logits: (B, 4*reg_max, H, W) —— DFL 之前的四边分布logits
        return:    (B, 1, H, W)          —— 质量分数 in [0,1]
        """
        B, C, H, W = reg_logits.shape
        rm = self.reg_max  # from Detect
        # (B, 4, reg_max, H, W) -> 概率
        dist = reg_logits.view(B, 4, rm, H, W).softmax(dim=2)

        # top-k 统计（按 GFLv2）：沿 reg_max 维取每条边的 topk 概率并拼接
        k = self.reg_topk
        vals, _ = torch.topk(dist, k, dim=2)  # (B,4,k,H,W)
        stats = [vals]  # [topk]

        if self.add_mean:
            mean = dist.mean(dim=2, keepdim=True)  # (B,4,1,H,W)
            stats.append(mean)

        # 拼成 conv 输入: (B, 4*(k + mean_flag), H, W)
        feat = torch.cat(stats, dim=2).reshape(B, 4*(k + (1 if self.add_mean else 0)), H, W)
        q = self.reg_conf[i](feat)  # (B,1,H,W) -> Sigmoid
        return q
    
    def forward(self, x):
        # ✅ 修复点：在循环外初始化，一次性收集所有层的 quality
        qualities = []

        for i in range(self.nl):
            xi = x[i]
            xi = self.stem[i](xi)
            if self.dat:     xi = self.dat[i](xi)
            if self.pos_cls: xi = self.pos_cls[i](xi)  # 简化：同一特征用于 cls/reg，两路可分也行
            if self.pos_reg: xi = self.pos_reg[i](xi)
            if self.cit_cls: xi = self.cit_cls[i](xi)
            if self.cit_reg: xi = self.cit_reg[i](xi)

            # 走 Ultralytics 的回归/分类塔
            box_i = self.cv2[i](xi)      # (B, 4*reg_max, H, W)
            cls_i = self.cv3[i](xi)      # (B, nc,        H, W)

            # ✅ 逐层 append，不要在循环内重置列表
            q_i = super()._compute_quality_from_logits(box_i, i, "one2many")
            qualities.append(q_i)

            x[i] = torch.cat((box_i, cls_i), 1)

        if self.training:
            return x

        # 推理：把 quality 乘到分类概率（复用父类带质量的解码）
        y = super()._inference_with_quality(x, qualities)
        return y if self.export else (y, x)


    def _compute_quality_from_logits(self, reg_logits: torch.Tensor, i: int) -> torch.Tensor:
        """
        reg_logits: (B, 4*reg_max, H, W) —— DFL 之前的四边分布 logits
        返回:      (B, 1, H, W)          —— 质量分数 I ∈ [0,1]
        机制：对每个边的 reg_max 桶做 softmax -> 取 top-k 概率（可加 mean）-> 1x1 head 输出 I
        """
        B, C, H, W = reg_logits.shape
        rm = self.reg_max
        dist = reg_logits.view(B, 4, rm, H, W).softmax(dim=2)  # (B,4,rm,H,W)

        k = getattr(self, 'reg_topk', 4)
        vals, _ = torch.topk(dist, k, dim=2)                   # (B,4,k,H,W)

        feats = [vals]
        if getattr(self, 'add_mean', True):
            feats.append(dist.mean(dim=2, keepdim=True))       # (B,4,1,H,W)

        feat = torch.cat(feats, dim=2).reshape(
            B, 4 * (k + (1 if getattr(self, 'add_mean', True) else 0)), H, W
        )                                                      # (B, 4*(k+mean), H, W)

        # reg_conf[i]: Conv/MLP -> (B,1,H,W)，末尾含 Sigmoid
        q = self.reg_conf[i](feat)
        return q

    @torch.no_grad()
    def _inference_with_quality(self, outs_list, qualities):
        """
        outs_list: List[(B, no, H, W)], qualities: List[(B,1,H,W)]
        目标：与 Detect._inference 对齐，但把 cls 概率乘以 I 后参与解码/NMS。
        如果你的 Detect 已有等价方法，可直接调用父类并在其中注入质量乘法。
        """
        # 1) 将每层的 (box_logits, cls_logits) 拆回
        reg = []
        cls = []
        for yi in outs_list:
            r, c = yi.split([4 * self.reg_max, self.nc], dim=1)
            reg.append(r)
            cls.append(c)

        # 2) 生成锚点/stride（沿用 Detect 的工具）
        # 下两行示例：你的 Detect 里通常有 make_anchors/_inference 的实现
        anchors, strides = self.make_anchors(reg)  # List -> (A,2), (A,)
        # 3) DFL 解码回 ltrb -> xyxy
        pd, ps = [], []
        for r, c, q in zip(reg, cls, qualities):
            B, _, H, W = c.shape
            # 分类概率（乘质量 I）
            cp = c.sigmoid() * q.clamp_(0, 1)      # (B,nc,H,W) * (B,1,H,W)
            ps.append(cp)

            # 回归分布 -> 期望（与 loss 的 decode 保持一致）
            dist = r.view(B, 4, self.reg_max, H, W).softmax(dim=2)
            proj = torch.arange(self.reg_max, device=r.device, dtype=r.dtype).view(1,1,self.reg_max,1,1)
            ltrb = (dist * proj).sum(dim=2)        # (B,4,H,W)
            pd.append(ltrb)

        # 4) 拼接各层 -> (B, A, nc)/(B, A, 4)，再用 dist2bbox/加 stride/偏移完成真实坐标解码
        # 这步通常由 Detect._inference 完成；此处仅示意：
        # boxes = self.dist2bbox(anchors, torch.cat(pd, dim=-1), xywh=False) * strides
        # scores = torch.cat(ps, dim=-1)
        # return self.postprocess(boxes, scores)  # NMS 等
        return self._inference_from_parts(pd, ps)  # 如果你已有内部工具，调用它

# GFLHeadv2_E2E(Detect)
class GFLHeadv2_E2E(Detect):
    # 假设 __init__ 里已经定义:
    # self.cv2[i]: 回归塔 -> (B, 4*reg_max, H, W)
    # self.cv3[i]: 分类塔 -> (B, nc,        H, W)
    # self.reg_conf[i]: DGQP 质量小头 (B, 4*(topk + mean_flag), H, W) -> (B,1,H,W)
    # self.reg_topk, self.add_mean, self.reg_max, self.nc, self.nl 等

    def forward(self, x):
        """
        x: List[Tensor], len = self.nl, 每层 (B, C, H, W)
        训练：返回每层 cat((box_i, cls_i), 1)，并将每层 quality 缓存到 self._qualities
        推理：基于 qualities 计算 J=C*I，走带质量的解码（与 GFLv2 一致）
        """
        B = x[0].shape[0]
        qualities = []    # ✅ 在循环外初始化，收集所有层的 quality
        outs = []

        for i in range(self.nl):
            xi = x[i]
            # 这些可选前端组件按你的实现保留（有则调用，无则跳过）
            if hasattr(self, 'stem')    and self.stem:    xi = self.stem[i](xi)
            if hasattr(self, 'dat')     and self.dat:     xi = self.dat[i](xi)
            if hasattr(self, 'pos_cls') and self.pos_cls: xi = self.pos_cls[i](xi)
            if hasattr(self, 'pos_reg') and self.pos_reg: xi = self.pos_reg[i](xi)
            if hasattr(self, 'cit_cls') and self.cit_cls: xi = self.cit_cls[i](xi)
            if hasattr(self, 'cit_reg') and self.cit_reg: xi = self.cit_reg[i](xi)

            # Ultralytics 风格：分头
            box_i = self.cv2[i](xi)     # (B, 4*reg_max, H, W) —— DFL logits
            cls_i = self.cv3[i](xi)     # (B, nc,        H, W) —— 分类 logits

            # ✅ 逐层质量 (DGQP)：从回归分布的 top-k (+ mean) 统计得到 I in [0,1]
            q_i = self._compute_quality_from_logits(box_i, i)
            qualities.append(q_i)        # (B,1,H,W)

            # 训练返回的张量约定：cat 后交给 loss 拆分
            outs.append(torch.cat((box_i, cls_i), 1))

        if self.training:
            # ✅ 训练端缓存，供 loss 端读取（构造 J 或仅 QFL 监督）
            self._qualities = qualities  # List[(B,1,H,W)]
            return outs

        # 🔎 推理端：把质量乘到分类概率，并按你的 Detect 解码路径输出
        y = self._inference_with_quality(outs, qualities)
        return y if getattr(self, 'export', False) else (y, outs)
