# -*- coding: utf-8 -*-
"""模型定义：1D CNN 提取局部时频特征 + LSTM 捕捉时序依赖"""
import torch
import torch.nn as nn


class FallNet(nn.Module):
    """CNN-LSTM 跌倒分类器
    输入: [batch, 6, 200]  (6通道 = 加速度x3 + 陀螺仪x3, 200个时间步)
    输出: [batch, 2]       (0=日常活动, 1=跌倒)
    """

    def __init__(self, in_channels=6, hidden=64, num_classes=2):
        super().__init__()

        # 温度缩放参数: post-hoc 校准用, 训练时固定为 1.0, 校准脚本单独优化
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # ---- CNN 部分: 三层卷积逐层提取特征, 每层后接BN+ReLU+池化 ----
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                      # 200 -> 100

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                      # 100 -> 50

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),                      # 50 -> 25
        )

        # ---- LSTM 部分: 输入128维CNN特征, 输出隐藏状态序列 ----
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden,
                            num_layers=1, batch_first=True)

        # ---- 分类头 ----
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        feat = self.cnn(x)              # [B, 128, 25]
        feat = feat.permute(0, 2, 1)    # -> [B, 25, 128] 适配LSTM(batch_first)
        out, _ = self.lstm(feat)        # [B, 25, hidden]
        last = out[:, -1, :]            # 取最后时刻的隐藏状态作为序列摘要
        logits = self.fc(last)          # [B, 2]
        # 温度缩放: logits / T, T>1 软化概率分布(降置信度), T<1 锐化
        return logits / self.temperature
