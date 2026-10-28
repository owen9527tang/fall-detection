# -*- coding: utf-8 -*-
"""方案B' 报警后静止验证
物理先验: 软跌倒(晕倒)后人体持续静止躺卧; 故意躺下/坐下的人稍后会再次活动
流程: 时序确认报警(阈值0.6/K=6) -> 对每个报警簇取报警后 D 秒的运动分数
      -> 均值 < 静止阈值 => 持续静止, 确认跌倒; 否则撤销该警报
运动分数: 窗口内3D加速度模长的时序标准差 + 陀螺仪模长标准差/100 (静止≈0, 运动明显>0.1)
"""
import numpy as np

from dataset import load_npz, split_by_subject
from temporal_confirm import trial_alarm

OUT_DIR = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs"
PROBS_PATH = OUT_DIR + r"\probs_cache.npz"

T, K = 0.6, 6        # 已选定的时序确认配置
STEP_SEC = 0.5
DURATIONS = [3.0, 5.0, 8.0]                  # 报警后观察时长(秒)
STILL_THS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30]  # 静止判定阈值


def movement_scores(x, chunk=20000):
    """每个窗口的运动分数 [N]: 加速度模长的时序标准差 + 陀螺仪模长标准差/100"""
    m = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), chunk):
        w = x[i:i + chunk]
        acc_mag = np.linalg.norm(w[:, :, 0:3], axis=2)   # [n,200] 加速度模长
        gyr_mag = np.linalg.norm(w[:, :, 3:6], axis=2)
        m[i:i + chunk] = acc_mag.std(axis=1) + gyr_mag.std(axis=1) / 100.0
    return m


def alarm_clusters(alarms):
    """把连续报警位置折叠成报警簇, 返回 (start, end) 对列表(end 不含)"""
    b = np.concatenate([[0], alarms.view(np.int8), [0]])
    idx = np.flatnonzero(np.diff(b))
    return idx.reshape(-1, 2)


def event_eval_still(probs, m, y, fid, t, k, dur_s, s_th):
    """带静止验证的事件级评估: 报警簇需通过"后续持续静止"验真"""
    w_follow = int(dur_s / STEP_SEC)          # 报警后观察的窗口数
    fall_total = fall_hits = fa_events = 0
    adl_seconds = 0.0
    for f in np.unique(fid):
        mask = fid == f
        p, mm = probs[mask], m[mask]
        detected = False
        for st, en in alarm_clusters(trial_alarm(p, t, k)):
            # 关键: 从报警簇【结束】之后开始观察, 避免把跌倒动作本身计入运动分数
            lo, hi = en, min(en + w_follow, len(mm))
            if hi <= lo:                       # 报警簇延伸到数据末尾, 无后续可查 -> 保守确认
                detected = True
                break
            if mm[lo:hi].mean() < s_th:        # 后续持续静止 -> 确认跌倒
                detected = True
                break
            # 被撤销, 继续检查下一个报警簇
        if y[mask][0] == 1:
            fall_total += 1
            fall_hits += detected
        else:
            fa_events += detected
            adl_seconds += mask.sum() * STEP_SEC
    fa_per_hour = fa_events / (adl_seconds / 3600) if adl_seconds else 0.0
    return fall_hits / max(fall_total, 1), fa_events, fa_per_hour


def main():
    cache = np.load(PROBS_PATH)
    x, y, subj, fid = load_npz()
    _, va, te = split_by_subject(subj)
    print("计算全量运动分数...")
    m = movement_scores(x)                     # 一次性计算, 所有组合复用
    del x
    m_val, m_test = m[va], m[te]               # 按划分切片, 与 probs/fid 对齐
    del m

    val_p, test_p = cache["val_p"], cache["test_p"]
    val_y, val_fid = y[va], fid[va]
    test_y, test_fid = y[te], fid[te]

    # ---------- 验证集扫参 ----------
    print("\n" + "=" * 66)
    print("验证集扫参 (报警后静止验证)")
    print("=" * 66)
    print(f"{'观察D(s)':>8s} {'静止阈值':>8s} {'事件召回':>10s} {'误报/小时':>10s}")
    results = []
    for dur in DURATIONS:
        for s in STILL_THS:
            rec, _, fah = event_eval_still(val_p, m_val, val_y, val_fid, T, K, dur, s)
            results.append((dur, s, rec, fah))
            print(f"{dur:8.1f} {s:8.2f} {rec:10.2%} {fah:10.1f}")
    results = np.array(results, dtype=[("d", float), ("s", float),
                                       ("rec", float), ("fah", float)])

    good = results[results["rec"] >= 0.95]
    if len(good):
        best = good[good["fah"].argmin()]
        rule = "召回>=95% 中误报最低"
    else:
        best = results[results["rec"].argmax()]
        rule = "无组合达95%, 取召回最高"
    print(f"\n选定配置: D={best['d']:.0f}s, 静止阈值={best['s']:.2f} ({rule})")
    print(f"  验证集: 事件召回 {best['rec']:.2%}, 误报 {best['fah']:.1f} 次/小时")

    # ---------- 测试集验证 ----------
    print("\n" + "=" * 66)
    print(f"测试集验证 (阈值{T}/K={K} + 静止验证 D={best['d']:.0f}s s={best['s']:.2f})")
    print("=" * 66)
    rec, n_fa, fah = event_eval_still(test_p, m_test, test_y, test_fid,
                                      T, K, best["d"], best["s"])
    print(f"  [纯时序确认基线] 事件召回 94.48% | 误报 43 个 | 12.2 次/小时")
    print(f"  [静止验证后      ] 事件召回 {rec:.2%} | 误报 {n_fa} 个 | {fah:.1f} 次/小时")


if __name__ == "__main__":
    main()
