# -*- coding: utf-8 -*-
"""时序确认扫参: 抑制孤立误报窗口, 要求连续 K 个窗口概率都超阈值才触发报警
流程: 推理缓存概率 -> 验证集扫(阈值,K)组合 -> 按目标选配置 -> 测试集验证
指标均为事件级(按试验聚合): 跌倒试验任一报警=检出, 日常试验任一报警=误报事件
"""
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader

from dataset import get_datasets, load_npz, split_by_subject
from model import FallNet

OUT_DIR = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs"
MODEL_PATH = OUT_DIR + r"\best_model.pt"
PROBS_PATH = OUT_DIR + r"\probs_cache.npz"

STEP_SEC = 0.5      # 滑窗步长
THRESHOLDS = np.arange(0.10, 0.61, 0.05)   # 候选决策阈值
KS = [1, 2, 3, 4, 5, 6]                    # 候选确认窗口数(连续K窗)
FA_H_TARGET = 10.0  # 误报目标: <=10 次/小时
RECALL_TARGET = 0.95


@torch.no_grad()
def predict_probs(model, loader, device):
    probs = []
    model.eval()
    for xb, _ in loader:
        probs.extend(torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.array(probs)


def trial_alarm(probs, threshold, k):
    """对一个试验的概率序列做时序确认: 连续 k 个窗口 >= 阈值 才算报警
    返回 [N] 的布尔数组(报警位置)"""
    if len(probs) < k:
        return np.zeros(len(probs), dtype=bool)
    ok = sliding_window_view(probs, k).min(axis=1) >= threshold  # 每个 K 窗组的最小值
    return np.concatenate([np.zeros(k - 1, dtype=bool), ok])     # 前 k-1 个位置不可能触发


def event_eval_simple(probs, y, fid, threshold, k):
    """事件级评估: 返回 (跌倒事件召回, 误报事件数, 每小时误报)"""
    fall_total = fall_hits = fa_events = 0
    adl_seconds = 0.0
    for f in np.unique(fid):
        mask = fid == f
        alarms = trial_alarm(probs[mask], threshold, k)
        if y[mask][0] == 1:
            fall_total += 1
            fall_hits += int(alarms.any())
        else:
            fa_events += int(alarms.any())
            adl_seconds += mask.sum() * STEP_SEC
    fa_per_hour = fa_events / (adl_seconds / 3600) if adl_seconds else 0.0
    return fall_hits / max(fall_total, 1), fa_events, fa_per_hour


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. 推理并缓存概率(重复扫参不再需要GPU) ----------
    try:
        cache = np.load(PROBS_PATH)
        val_p, test_p = cache["val_p"], cache["test_p"]
        print("已加载概率缓存")
    except FileNotFoundError:
        model = FallNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        _, val_set, test_set = get_datasets()
        print("推理验证集...")
        val_p = predict_probs(model, DataLoader(val_set, batch_size=256), device)
        print("推理测试集...")
        test_p = predict_probs(model, DataLoader(test_set, batch_size=256), device)
        np.savez_compressed(PROBS_PATH, val_p=val_p, test_p=test_p)
        print(f"概率已缓存至 {PROBS_PATH}")

    x, y, subj, fid = load_npz()
    _, va, te = split_by_subject(subj)
    val_y, val_fid = y[va], fid[va]
    test_y, test_fid = y[te], fid[te]

    # ---------- 2. 验证集扫参: (阈值, K) 网格 ----------
    print("\n" + "=" * 66)
    print(f"验证集扫参 (目标: 事件召回 >= {RECALL_TARGET:.0%} 且 误报 <= {FA_H_TARGET:.0f}次/小时)")
    print("=" * 66)
    print(f"{'阈值':>6s} {'K':>3s} {'事件召回':>10s} {'误报/小时':>10s}")
    results = []
    for t in THRESHOLDS:
        for k in KS:
            rec, _, fah = event_eval_simple(val_p, val_y, val_fid, t, k)
            results.append((float(t), k, rec, fah))
            print(f"{t:6.2f} {k:3d} {rec:10.2%} {fah:10.1f}")
    results = np.array(results, dtype=[("t", float), ("k", int), ("rec", float), ("fah", float)])

    # 选择规则: 满足误报目标中召回最高; 无满足者则放宽到误报最小
    good = results[results["fah"] <= FA_H_TARGET]
    if len(good):
        best = good[good["rec"].argmax()]
        rule = f"误报<={FA_H_TARGET:.0f}/h 内召回最高"
    else:
        best = results[results["fah"].argmin()]
        rule = "无组合满足目标, 取误报最低"
    print(f"\n选定配置: 阈值={best['t']:.2f}, K={best['k']} ({rule})")
    print(f"  验证集: 事件召回 {best['rec']:.2%}, 误报 {best['fah']:.1f} 次/小时")

    # ---------- 3. 测试集验证 ----------
    print("\n" + "=" * 66)
    print(f"测试集验证 (阈值={best['t']:.2f}, K={best['k']})")
    print("=" * 66)
    for tag, (p_, y_, f_) in [("默认0.5/K=1", (test_p, test_y, test_fid))]:
        rec, n_fa, fah = event_eval_simple(p_, y_, f_, 0.5, 1)
        print(f"  [{tag:14s}] 事件召回 {rec:.2%} | 误报 {n_fa} 个 | {fah:.1f} 次/小时")
    rec, n_fa, fah = event_eval_simple(test_p, test_y, test_fid, best["t"], int(best["k"]))
    print(f"  [选定配置      ] 事件召回 {rec:.2%} | 误报 {n_fa} 个 | {fah:.1f} 次/小时")

    # 附加: 测试集上不同K的敏感性(固定选定阈值)
    print("\n测试集 K 敏感性 (固定阈值):")
    for k in KS:
        rec, _, fah = event_eval_simple(test_p, test_y, test_fid, best["t"], k)
        print(f"  K={k}: 事件召回 {rec:.2%} | 误报 {fah:.1f} 次/小时")


if __name__ == "__main__":
    main()
