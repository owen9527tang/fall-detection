# -*- coding: utf-8 -*-
"""SisFall 预处理：读取原始txt -> 物理单位换算 -> 滑动窗口切片 -> 保存npz"""
import io
import os
import re

import numpy as np
import pandas as pd

# ============ 配置（按实际路径修改） ============
DATA_DIR = r"C:\Users\ASUS\Desktop\fall database\archive (1)\SisFall_dataset"  # 完整版: SA01-SA23青年 / SE01-SE15老年
OUT_PATH = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs\sisfall_full_windows.npz"

FS = 200            # SisFall 采样频率 200Hz
WIN_SEC = 1.0       # 窗口时长 1 秒（跌倒是短时事件，1s 窗口是常用配置）
OVERLAP = 0.5       # 窗口重叠 50%，增加样本量且保证边界事件不被切断

WIN_SIZE = int(FS * WIN_SEC)                # 每窗 200 个采样点
STEP_SIZE = int(WIN_SIZE * (1 - OVERLAP))   # 滑动步长 100

# 传感器原始值 -> 物理单位换算系数
ACC_SCALE = 32.0 / 8192.0      # ADXL345 ±16g、13位有符号数 -> 单位 g
GYRO_SCALE = 4000.0 / 16384.0  # ITG3200 ±2000°/s、14位 -> 单位 °/s

FALL_PATTERN = re.compile(r"^F\d+_")   # 文件名以 F 开头 => 跌倒样本
DATA_PATTERN = re.compile(r"^[DF]\d+_[A-Z]+\d+_R\d+\.txt$",
                          re.IGNORECASE)  # 合法数据文件: 如 D01_SA01_R01.txt / F13_SE06_R03.txt


def read_sensor_file(path):
    """读取单个 txt，返回 [样本数, 6] 数组（单位：g 和 °/s）
    本数据每行 9 列、逗号分隔、行尾带分号（如 " 17,-179,...,-279;"），
    只取前 6 列：0-2 加速度、3-5 陀螺仪。"""
    with open(path, "r") as f:
        text = f.read().replace(";", "")   # 去掉行尾分号，否则无法解析
    raw = pd.read_csv(io.StringIO(text), header=None).values
    acc = raw[:, 0:3] * ACC_SCALE                 # 前3列：加速度
    if raw.shape[1] >= 6:
        gyro = raw[:, 3:6] * GYRO_SCALE           # 4-6列：陀螺仪
        return np.hstack([acc, gyro])             # 6 通道：acc + gyro
    return acc                                    # 兼容仅含加速度的文件


def sliding_windows(sig, win, step):
    """滑动窗口切片，返回 [N, win, C]，用索引矩阵避免 Python 循环"""
    n = (len(sig) - win) // step + 1
    if n <= 0:
        return np.empty((0, win, sig.shape[1]))
    idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]
    return sig[idx]


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    xs, ys, subs, fids = [], [], [], []
    files = []
    for root, _, names in os.walk(DATA_DIR):        # 递归遍历 ADL/、FALL/ 子目录
        for name in names:
            if name.lower().endswith(".txt") and DATA_PATTERN.match(name):
                files.append(os.path.join(root, name))  # 跳过 Readme.txt 等非数据文件
    files = sorted(files)
    n_fall = sum(1 for p in files if FALL_PATTERN.match(os.path.basename(p)))
    print(f"共发现 {len(files)} 个数据文件 (跌倒 {n_fall} 个, 日常 {len(files) - n_fall} 个)")

    for i, path in enumerate(files, 1):
        fname = os.path.basename(path)
        label = 1 if FALL_PATTERN.match(fname) else 0   # 1=跌倒, 0=日常活动
        subj = fname.split("_")[1]                      # 受试者编号, 如 SA01
        sig = read_sensor_file(path)
        if len(sig) < WIN_SIZE:
            continue
        wins = sliding_windows(sig, WIN_SIZE, STEP_SIZE)
        xs.append(wins)
        ys.append(np.full(len(wins), label, dtype=np.int64))
        subs.append(np.full(len(wins), subj))
        fids.append(np.full(len(wins), i - 1, dtype=np.int32))  # 试验(文件)ID: 事件级评估用
        if i % 200 == 0:
            print(f"进度: {i}/{len(files)}")

    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    subj = np.concatenate(subs)
    fid = np.concatenate(fids)
    np.savez_compressed(OUT_PATH, x=x, y=y, subj=subj, fid=fid)
    print(f"完成! 共 {len(x)} 个窗口, 其中跌倒窗口 {int(y.sum())} 个 "
          f"({y.mean() * 100:.1f}%), 已保存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
