# -*- coding: utf-8 -*-
"""温度缩放校准: 在验证集上学习标量温度 T, 最小化 NLL, 修复 Focal Loss 导致的过度自信

原理:
  logits_scaled = logits / T
  - T > 1: 软化概率分布(降低置信度), 适合校准过度自信的模型
  - T < 1: 锐化概率分布(提高置信度)
  T 通过最小化验证集的负对数似然(NLL)学习, 是 post-hoc 校准的标准方法.

注意: 只优化 temperature 一个标量, 其余参数全部冻结, 不会影响模型判别能力.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import FallNet

MODEL_PATH = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs\best_model.pt"


@torch.no_grad()
def collect_val_logits(model, loader, device):
    """收集验证集的 logits 和标签, 用于温度优化"""
    model.eval()
    all_logits, all_labels = [], []
    for xb, yb in loader:
        # 用 T=1.0 收集原始 logits (temperature 是 Parameter, 直接除以它)
        with torch.no_grad():
            raw = model(xb.to(device)) * model.temperature
        all_logits.append(raw.cpu())
        all_labels.append(yb)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallNet().to(device)
    # 旧模型 state_dict 无 temperature, strict=False 让其保持初始值 1.0
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)

    _, val_set, _ = get_datasets()
    val_loader = DataLoader(val_set, batch_size=256, pin_memory=True)

    # 收集验证集原始 logits (T=1.0 时的输出)
    val_logits, val_labels = collect_val_logits(model, val_loader, device)
    val_logits = val_logits.to(device)
    val_labels = val_labels.to(device)

    # 冻结全部参数, 只放开 temperature
    for p in model.parameters():
        p.requires_grad_(False)
    model.temperature.requires_grad_(True)

    # LBFGS 优化 NLL (温度缩放的标准优化器, 对单参数问题收敛快且稳定)
    optimizer = torch.optim.LBFGS([model.temperature], lr=0.1, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    before_nll = criterion(val_logits / model.temperature.item(), val_labels).item()
    print(f"优化前 T={model.temperature.item():.4f} | 验证集 NLL={before_nll:.4f}")

    def closure():
        optimizer.zero_grad()
        loss = criterion(val_logits / model.temperature, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    t_opt = model.temperature.item()
    after_nll = criterion(val_logits / t_opt, val_labels).item()
    print(f"优化后 T={t_opt:.4f} | 验证集 NLL={after_nll:.4f}")

    # 保存校准后的模型 (temperature 已写入 state_dict, evaluate.py 会自动用)
    model.temperature.requires_grad_(False)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"已保存校准后模型至 {MODEL_PATH}")

    # 快速验证测试集 ECE 变化
    from evaluate import reliability_diagram, predict_probs
    _, _, test_set = get_datasets()
    test_loader = DataLoader(test_set, batch_size=256, pin_memory=True)
    test_p = predict_probs(model, test_loader, device)
    test_y_true = test_set.y.numpy()
    _, _, ece = reliability_diagram(test_y_true, test_p)
    print(f"测试集 ECE = {ece:.4f}  (校准目标: 越小越接近0)")


if __name__ == "__main__":
    main()
