"""
街景图像数据加载器

两种模式:
    Picdataset_Realtime (rt): 按需从 LevelDB 读取全景图，用 LRU 缓存最近 N 张
    Picdataset_Cache   (cc): 启动时预加载所有全景图，用 ResNet18 提取特征后缓存
"""

import torch
import streetlearn_pb2
import cv2
from torch.utils.data import Dataset
import numpy as np
import plyvel

import torchvision as tov
import pytorch360convert
from torchvision import models, transforms as tt
from collections import OrderedDict


# ============================================================
# 图像预处理（ImageNet 标准化）
# ============================================================
IMG_SIZE = 84
IMG_NORMALIZE = tt.Compose([
    tt.Resize((IMG_SIZE, IMG_SIZE)),
    tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class StandardScaler:
    """数据标准化（用于 pics_buff 的归一化）"""
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        return (data - self.mean) / self.std


class Picdataset_Realtime:
    """实时模式：按需加载 + LRU 缓存"""

    def __init__(self, args):
        self.db_root = args.db_root_h
        self.cap = args.max_buf
        self.cache_root = args.cache_root
        self.device = args.device
        self.pixel = args.pixel
        self.infos = np.load(f"{self.cache_root}/static_info.npy", allow_pickle=True)
        self.neig = np.load(f"{self.cache_root}/neig_info.npy", allow_pickle=True)

        self.buf = OrderedDict()
        self.pano = streetlearn_pb2.Pano()
        self.scaler = StandardScaler()

    def scaled_view(self, frame, cur_head):
        view = pytorch360convert.e2p(frame, fov_deg=120, h_deg=cur_head, v_deg=0,
                                       out_hw=(self.pixel, self.pixel))
        view = IMG_NORMALIZE(view)
        return view

    def load_pano(self, pano_id):
        db = plyvel.DB(self.db_root, create_if_missing=False)
        for ke in pano_id:
            self.pano.ParseFromString(db.get(ke.encode('utf-8')))
            frame = tov.io.decode_image(
                torch.frombuffer(self.pano.compressed_image, dtype=torch.uint8),
                mode=tov.io.ImageReadMode.RGB
            )
            self.buf[ke] = (frame / 255).to(self.device)
            if len(self.buf) >= self.cap:
                self.buf.popitem(last=False)
        db.close()

    def update_buff(self, pano_id):
        new_id = [p for p in pano_id if p not in self.buf]
        if new_id:
            self.load_pano(new_id)

    def get_(self, pano_id, cur_heading):
        self.update_buff(pano_id)
        frames = [self.buf[i] for i in pano_id]
        views_ = [self.scaled_view(f, cur_heading[i]) for i, f in enumerate(frames)]
        return torch.stack(views_, dim=0)


class Picdataset_Cache:
    """缓存模式：启动时全量提取 ResNet18 特征，后续直接查表"""

    def __init__(self, args):
        self.args = args
        self.db_root = args.db_root_h
        self.cache_root = args.cache_root
        self.device = args.device
        self.le = args.le
        self.pixel = args.pixel

        self.infos = np.load(f"{self.cache_root}/static_info.npy", allow_pickle=True)
        if self.le is not None:
            area_index = np.load(f"{self.cache_root}/area_index.npy")
            self.infos = self.infos[area_index]

        # 加载 ResNet18 并放到正确设备
        self.cnn = models.resnet18(weights='IMAGENET1K_V1')
        if torch.cuda.is_available():
            self.cnn = self.cnn.to(args.device)
        self.cnn.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.pano = streetlearn_pb2.Pano()
        self.scaler = StandardScaler()
        self.pics_buff = None
        self.initial_buff()

    def initial_buff(self):
        print("Pre-extracting image features with ResNet18...")
        pano_id = self.infos[:, 0]
        pics = []
        db = plyvel.DB(self.db_root, create_if_missing=False)
        for ke in pano_id:
            self.pano.ParseFromString(db.get(ke.encode('utf-8')))
            frame = tov.io.decode_image(
                torch.frombuffer(self.pano.compressed_image, dtype=torch.uint8),
                mode=tov.io.ImageReadMode.RGB
            )
            frame = frame.float() / 255
            view = self.scaler.transform(frame) if hasattr(self.scaler, 'transform') else frame
            # 注意：这里不使用 e2p 投影（原代码注释掉了），直接用原图
            pics.append(view)
        db.close()

        self.pics_buff = torch.stack(pics).to(self.device)
        with torch.no_grad():
            self.pics_buff = self.cnn(self.pics_buff)
        print(f"Feature cache ready: {self.pics_buff.shape}")

    def get_(self, pano_id, cur_heading):
        return self.pics_buff[pano_id]
