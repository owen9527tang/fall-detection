# -*- coding: utf-8 -*-
"""训练脚本：加权交叉熵处理类别不平衡 + 早停保存最优模型"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_datasets, SEED
from model import FallNet


class FocalLoss(nn.Module):
    """Focal Loss: 在交叉熵基础上降低易分样本权重, 让模型聚焦难分样本

    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
      - p_t   : 正确类别的预测概率
      - γ(gamma): 聚焦参数, 越大越抑制易分样本 (论文默认 2.0)
      - α(alpha): 类别平衡因子, 沿用加权交叉熵的类别权重即可

    相比加权交叉熵的优势: 日常活动(多数类)样本虽多但易分, 其梯度会被
    (1-p_t)^γ 大幅压低; 跌倒(少数类)难分样本的梯度被保留, 从而提升
    对少数类的判别力.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha          # [num_classes] 的类别权重, None 表示不加权
        self.gamma = gamma

    def forward(self, logits, target):
        # reduction='none' 保留每个样本的 CE, 便于后续逐样本乘聚焦权重
        ce = nn.functional.cross_entropy(logits, target,
                                         weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)          # p_t = 正确类别的预测概率
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()

# ============ 配置 ============
BATCH_SIZE = 128
EPOCHS = 60            # 有在线增强后收敛更慢，适当加长轮数
LR = 1e-3
PATIENCE = 12           # 验证损失连续 12 轮不下降则早停
MODEL_PATH = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs\best_model.pt"


def main():
    np.random.seed(SEED)          # 固定随机种子，保证增强可复现
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 数据 ----
    train_set, val_set, _ = get_datasets()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, pin_memory=True)

    # ---- 类别不平衡: 权重 = 1 / 类别频率, 归一化到平均为1 ----
    counts = np.bincount(train_set.y.numpy())
    weights = counts.sum() / (2.0 * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"类别权重(日常/跌倒): {weights.tolist()}")

    # ---- 模型 / 损失 / 优化器 / 学习率调度 ----
    model = FallNet().to(device)
    # Focal Loss: 保留类别权重(alpha)处理不平衡, 额外用 gamma=2.0 抑制易分样本
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    # 排除 temperature 参数: 训练阶段固定 T=1.0, 留待校准脚本 post-hoc 优化
    trainable_params = [p for n, p in model.named_parameters() if n != "temperature"]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    # 验证损失停滞 3 轮 -> 学习率减半，帮助后期精调
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val, wait = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        # ===== 训练一轮 =====
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        # ===== 验证 =====
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()

        train_loss = total_loss / len(train_set)
        val_loss /= len(val_set)
        val_acc = correct / len(val_set)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d} | 训练损失 {train_loss:.4f} | "
              f"验证损失 {val_loss:.4f} | 验证准确率 {val_acc:.4f} | lr {lr_now:.2e}")
        scheduler.step(val_loss)

        # ===== 早停 + 保存最优 =====
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> 保存最优模型")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"验证损失 {PATIENCE} 轮未改善, 提前停止训练")
                break

    print(f"训练结束, 最优验证损失 {best_val:.4f}, 模型已保存至 {MODEL_PATH}")


if __name__ == "__main__":
    main()
