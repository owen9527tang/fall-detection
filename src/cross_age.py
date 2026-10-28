# -*- coding: utf-8 -*-
"""阶段D 跨年龄泛化实验
D1: 仅青年(SA)训练 -> 老年(SE)测试
D2: 混合(SA+SE训练组)训练 -> 同一老年(SE)测试组
测试集固定为 主实验测试组中的 SE 受试者(含唯一模拟跌倒的SE06), 两组完全一致
评估: 窗口级指标(@0.5/@0.25) + 完整三级流水线事件级(阈值0.6/K=6/静止验真3s)
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FallDataset, load_npz, SEED
from model import FallNet
from temporal_confirm import predict_probs, trial_alarm
from stillness_verify import movement_scores, alarm_clusters

OUT_DIR = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs"
BATCH_SIZE = 128
EPOCHS, LR, PATIENCE = 60, 1e-3, 12
STEP_SEC = 0.5
PIPE = dict(t=0.6, k=6, dur=3.0, s_th=0.02)   # 已定型的三级流水线参数


def subject_sets(subj):
    """复现 dataset.split_by_subject 的受试者划分(相同种子/顺序)"""
    rng = np.random.RandomState(SEED)
    subjects = np.unique(subj)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    return subjects[:n_train], subjects[n_train:n_train + n_val], subjects[n_train + n_val:]


def windows_of(subj, subjects):
    """取属于给定受试者集合的所有窗口下标"""
    return np.where(np.isin(subj, subjects))[0]


def train_model(tag, x_tr, y_tr, x_va, y_va, model_path):
    """与 train.py 完全一致的训练流程(加权CE/Adam/调度器/早停); 支持断点续跑"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path):        # 断点续跑: 已有模型则跳过训练
        stats = FallDataset(x_tr, y_tr)   # 仅复算该训练集的 mean/std
        print(f"[{tag}] 发现已保存模型, 跳过训练")
        return stats.mean, stats.std
    train_set = FallDataset(x_tr, y_tr, augment=True)
    val_set = FallDataset(x_va, y_va, train_set.mean, train_set.std)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, pin_memory=True)

    counts = np.bincount(train_set.y.numpy())
    weights = torch.tensor(counts.sum() / (2.0 * counts),
                           dtype=torch.float32, device=device)
    print(f"[{tag}] 训练 {len(train_set)} / 验证 {len(val_set)} 窗口, "
          f"类别权重 {np.round(weights.cpu().numpy(), 3).tolist()}")

    model = FallNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val, wait = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_set)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            print(f"[{tag}] Epoch {epoch:2d} | 验证损失 {val_loss:.4f} (最优 {best_val:.4f})")
            if wait >= PATIENCE:
                print(f"[{tag}] 早停于第 {epoch} 轮, 最优验证损失 {best_val:.4f}")
                break
    return train_set.mean, train_set.std


def window_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return (dict(rec=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                 prec=tp / max(tp + fp, 1), acc=(tp + tn) / len(y)))


def pipeline_event_eval(p, m, y, fid, t, k, dur, s_th):
    """三级流水线事件级评估(仅老年测试集)"""
    w_follow = int(dur / STEP_SEC)
    fall_total = fall_hits = fa = 0
    adl_seconds = 0.0
    for f in np.unique(fid):
        mask = fid == f
        detected = False
        for st, en in alarm_clusters(trial_alarm(p[mask], t, k)):
            lo, hi = en, min(en + w_follow, len(m[mask]))
            if hi <= lo or m[mask][lo:hi].mean() < s_th:
                detected = True
                break
        if y[mask][0] == 1:
            fall_total += 1
            fall_hits += detected
        else:
            fa += detected
            adl_seconds += mask.sum() * STEP_SEC
    return fall_hits / max(fall_total, 1), fa, fa / (adl_seconds / 3600) if adl_seconds else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, subj, fid = load_npz()
    train_s, val_s, test_s = subject_sets(subj)

    sa_all = windows_of(subj, [s for s in np.unique(subj) if s.startswith("SA")])
    se_train = windows_of(subj, [s for s in train_s if s.startswith("SE")])
    sa_val = windows_of(subj, [s for s in val_s if s.startswith("SA")])
    se_val = windows_of(subj, [s for s in val_s if s.startswith("SE")])
    # 固定的老年测试组: 测试组中的 SE 受试者
    te_se_subjects = [s for s in test_s if s.startswith("SE")]
    te_idx = windows_of(subj, te_se_subjects)
    x_te, y_te, subj_te, fid_te = x[te_idx], y[te_idx], subj[te_idx], fid[te_idx]
    m_te = movement_scores(x_te)   # 老年测试集运动分数(两组共用)
    n_fall_trials = len(np.unique(fid_te[y_te == 1]))
    print(f"老年测试组: 受试者 {te_se_subjects}, {len(te_idx)} 窗口, "
          f"跌倒试验 {n_fall_trials} 个")
    print("(注意: 该数据集仅 SE06 模拟了老年人跌倒)\n")

    results = {}
    # ---------- D1: 仅青年训练 ----------
    mean1, std1 = train_model("D1仅青年", x[sa_all], y[sa_all],
                              x[sa_val], y[sa_val], OUT_DIR + r"\model_D1.pt")
    model = FallNet().to(device)
    model.load_state_dict(torch.load(OUT_DIR + r"\model_D1.pt", map_location=device))
    ds = FallDataset(x_te, y_te, mean1, std1)
    p1 = predict_probs(model, DataLoader(ds, batch_size=256), device)
    results["D1 仅青年训练"] = (p1, y_te)

    # ---------- D2: 混合训练 ----------
    d2_train = np.concatenate([sa_all, se_train])
    d2_val = np.concatenate([sa_val, se_val])
    mean2, std2 = train_model("D2混合", x[d2_train], y[d2_train],
                              x[d2_val], y[d2_val], OUT_DIR + r"\model_D2.pt")
    model = FallNet().to(device)
    model.load_state_dict(torch.load(OUT_DIR + r"\model_D2.pt", map_location=device))
    ds = FallDataset(x_te, y_te, mean2, std2)
    p2 = predict_probs(model, DataLoader(ds, batch_size=256), device)
    results["D2 混合训练"] = (p2, y_te)

    # ---------- 汇总对比 ----------
    print("\n" + "=" * 70)
    print("跨年龄泛化对比 (同一老年测试组)")
    print("=" * 70)
    print(f"{'模型':14s} {'Rec@0.5':>8s} {'Spec@0.5':>9s} {'Rec@0.25':>9s} "
          f"{'事件召回':>9s} {'误报/小时':>10s}")
    for tag, (p, _) in results.items():
        w5 = window_metrics(y_te, p, 0.5)
        w25 = window_metrics(y_te, p, 0.25)
        rec, n_fa, fah = pipeline_event_eval(p, m_te, y_te, fid_te, **PIPE)
        print(f"{tag:14s} {w5['rec']:8.2%} {w5['spec']:9.2%} {w25['rec']:9.2%} "
              f"{rec:9.2%} {fah:9.1f} ({n_fa}个)")
    print("\n结论要点: 混合训练若在老年组上召回/误报更优, 说明加入老年数据改善跨年龄保护")


if __name__ == "__main__":
    main()
