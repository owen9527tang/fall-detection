# -*- coding: utf-8 -*-
"""数据集封装：加载npz -> 按受试者划分 -> 在线数据增强(仅训练集) -> 标准化 -> PyTorch Dataset"""
import numpy as np
import torch
from torch.utils.data import Dataset

NPZ_PATH = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs\sisfall_full_windows.npz"

SEED = 42
# 按受试者划分比例：训练 60% / 验证 20% / 测试 20%
SPLIT_RATIO = (0.6, 0.2, 0.2)

# ============ 数据增强配置（均为物理单位：g / °/s，仅作用于训练集） ============
AUG_PROB = 0.5                 # 每种变换独立施加的概率
NOISE_SIGMA = np.array([0.05, 0.05, 0.05, 2.0, 2.0, 2.0])  # 加噪标准差: 前3通道acc(g), 后3通道gyro(°/s)
ROT_MAX_DEG = 30.0             # 随机旋转的最大角度（模拟佩戴角度差异）
SCALE_RANGE = (0.85, 1.15)     # 时间缩放范围（模拟动作快慢差异）


def load_npz(path=NPZ_PATH):
    """读取预处理生成的 npz 文件（fid 为试验/文件ID，用于事件级评估）"""
    data = np.load(path, allow_pickle=True)
    return data["x"], data["y"], data["subj"], data["fid"]


def split_by_subject(subj):
    """按受试者ID划分数据集，避免同一人的数据同时出现在训练和测试集（防止数据泄漏）"""
    rng = np.random.RandomState(SEED)
    subjects = np.unique(subj)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])
    train_s = subjects[:n_train]
    val_s = subjects[n_train:n_train + n_val]

    train_idx = np.where(np.isin(subj, train_s))[0]
    val_idx = np.where(np.isin(subj, val_s))[0]
    test_idx = np.where(~np.isin(subj, np.concatenate([train_s, val_s])))[0]
    return train_idx, val_idx, test_idx


# ============ 三种在线增强（保标签变换：加噪 / 随机旋转 / 时间缩放） ============

def _jitter(w):
    """叠加高斯噪声，模拟不同设备的传感器噪声差异"""
    noise = np.random.normal(0.0, NOISE_SIGMA[:, None], w.shape)
    return (w + noise).astype(np.float32)


def _random_rotation(w):
    """整体随机旋转，模拟传感器佩戴角度不同（acc和gyro用同一旋转矩阵，保持3D几何一致）"""
    axis = np.random.randn(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return w
    axis = (axis / norm).astype(np.float32)
    ang = np.deg2rad(np.random.uniform(0, ROT_MAX_DEG))
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)  # Rodrigues公式
    w = w.copy()
    w[0:3] = R @ w[0:3]   # [3,3]@[3,T]: 每个时间步的3D矢量同旋转
    w[3:6] = R @ w[3:6]
    return w


def _time_scale(w):
    """时间轴随机缩放后线性插值回原长，模拟不同人动作快慢差异"""
    s = np.random.uniform(*SCALE_RANGE)
    C, T = w.shape
    new_len = max(8, int(round(T * s)))
    new_t = np.linspace(0, T - 1, new_len)
    old_t = np.arange(T)
    out = np.empty((C, T), dtype=np.float32)
    fit = min(new_len, T)          # 拉长时截断到T点, 缩短时取全部新点
    for c in range(C):
        out[c, :fit] = np.interp(new_t[:fit], old_t, w[c])
    if new_len < T:                # 缩短 -> 时间轴末尾补零
        out[:, new_len:] = 0.0
    return out


def apply_augmentation(w):
    """对单个窗口 [C, T] 依次以 50% 概率施加三种变换"""
    if np.random.rand() < AUG_PROB:
        w = _jitter(w)
    if np.random.rand() < AUG_PROB:
        w = _random_rotation(w)
    if np.random.rand() < AUG_PROB:
        w = _time_scale(w)
    return w


class FallDataset(Dataset):
    """跌倒检测数据集：原始数据存为 [N, C, T]，增强在取样本时在线进行，随后标准化"""

    def __init__(self, x, y, mean=None, std=None, augment=False):
        # 标准化统计量：训练集计算自己的 mean/std，验证/测试集必须复用训练集的
        if mean is None:
            self.mean = x.mean(axis=(0, 1), keepdims=True)  # [1,1,C] 按通道
            self.std = x.std(axis=(0, 1), keepdims=True) + 1e-8
        else:
            self.mean, self.std = mean, std
        # 存原始物理值 [N, C, T]，增强需要在物理单位空间进行
        self.x = torch.tensor(np.ascontiguousarray(x.transpose(0, 2, 1)), dtype=torch.float32)
        self.mean_t = torch.tensor(self.mean.reshape(-1, 1), dtype=torch.float32)  # [C,1]
        self.std_t = torch.tensor(self.std.reshape(-1, 1), dtype=torch.float32)    # [C,1]
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        w = self.x[idx]                       # [C, T] 物理单位
        if self.augment:
            w = torch.from_numpy(apply_augmentation(w.numpy()))
        w = (w - self.mean_t) / self.std_t    # 标准化放在增强之后
        return w, self.y[idx]


def get_datasets(npz_path=NPZ_PATH):
    """一步到位：返回 训练集(含增强)/验证集/测试集 三个 Dataset 对象"""
    x, y, subj, _ = load_npz(npz_path)   # 训练/验证/测试只需 x/y/subj，fid 留给评估脚本
    tr, va, te = split_by_subject(subj)
    train_set = FallDataset(x[tr], y[tr], augment=True)      # 仅训练集开启在线增强
    val_set = FallDataset(x[va], y[va], train_set.mean, train_set.std)
    test_set = FallDataset(x[te], y[te], train_set.mean, train_set.std)
    print(f"数据集划分(按受试者): 训练 {len(train_set)} / "
          f"验证 {len(val_set)} / 测试 {len(test_set)} 个窗口 (训练集在线增强已开启)")
    return train_set, val_set, test_set
