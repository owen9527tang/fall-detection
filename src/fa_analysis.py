# -*- coding: utf-8 -*-
"""误报来源分析(方案A): 选定配置(阈值0.6/K=6)在测试集上
1) 哪些ADL活动贡献了误报事件 -> FA数 + 该活动的FA率(次/小时)
2) 漏检的跌倒是哪些类型(F代码) / 哪个年龄段
3) 误报事件按年龄段分布
概率取自 probs_cache.npz, 无需重新推理
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_npz, split_by_subject
from temporal_confirm import trial_alarm

DATA_DIR = r"C:\Users\ASUS\Desktop\fall database\archive (1)\SisFall_dataset"
OUT_DIR = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs"
PROBS_PATH = OUT_DIR + r"\probs_cache.npz"
FIG_PATH = OUT_DIR + r"\fa_sources.png"

T, K = 0.6, 6      # 选定的时序确认配置
STEP_SEC = 0.5
DATA_PATTERN = re.compile(r"^[DF]\d+_[A-Z]+\d+_R\d+\.txt$", re.IGNORECASE)

# SisFall 官方活动对照表(中译)
ACT_DESC = {
    "D01": "慢走", "D02": "快走", "D03": "慢跑", "D04": "快跑",
    "D05": "上下楼梯(慢)", "D06": "上下楼梯(快)",
    "D07": "慢坐半高椅再起身", "D08": "快坐半高椅再起身",
    "D09": "慢坐矮椅再起身", "D10": "快坐矮椅再起身",
    "D11": "试图起身但瘫坐回椅", "D12": "坐下慢躺再坐起",
    "D13": "坐下快躺再坐起", "D14": "仰卧翻身侧躺再躺回",
    "D15": "站立屈膝慢蹲再起身", "D16": "站立弯腰(不屈膝)再起身",
    "D17": "进出汽车", "D18": "走路被绊", "D19": "轻跳够高处物体",
    "F01": "走路滑倒(向前)", "F02": "走路滑倒(向后)", "F03": "走路滑倒(侧向)",
    "F04": "走路绊倒(向前)", "F05": "慢跑绊倒(向前)", "F06": "走路晕倒(垂直)",
    "F07": "走路晕倒(手撑缓冲)", "F08": "起身时向前跌", "F09": "起身时侧向跌",
    "F10": "坐下时向前跌", "F11": "坐下时向后跌", "F12": "坐下时侧向跌",
    "F13": "坐着向前晕倒", "F14": "坐着向后晕倒", "F15": "坐着侧向晕倒",
}


def rebuild_files():
    """按 preprocess 相同规则重建文件列表(fid = 在此列表中的下标)"""
    files = []
    for root, _, names in os.walk(DATA_DIR):
        for name in names:
            if name.lower().endswith(".txt") and DATA_PATTERN.match(name):
                files.append(os.path.join(root, name))
    return sorted(files)


def main():
    cache = np.load(PROBS_PATH)
    x, y, subj, fid = load_npz()
    _, _, te = split_by_subject(subj)
    test_p, test_y, test_fid = cache["test_p"], y[te], fid[te]

    files = rebuild_files()
    acts = np.array([os.path.basename(f).split("_")[0] for f in files])
    f_subj = np.array([os.path.basename(f).split("_")[1] for f in files])
    print(f"文件列表重建: {len(files)} 个, fid 最大值 {test_fid.max()} (一致性校验)")

    # ---------- 按试验聚合报警 ----------
    fa_acts, miss_acts = [], []          # 误报活动代码 / 漏检跌倒活动代码
    fa_by_age, fall_trial_by_age = {}, {}  # 年龄段统计
    n_fall = 0
    for f in np.unique(test_fid):
        mask = test_fid == f
        alarmed = trial_alarm(test_p[mask], T, K).any()
        lab, act, s = test_y[mask][0], acts[f], f_subj[f]
        age = "老年SE" if s.startswith("SE") else "青年SA"
        if lab == 1:
            n_fall += 1
            fall_trial_by_age[age] = fall_trial_by_age.get(age, 0) + 1
            if not alarmed:
                miss_acts.append(act)
        elif alarmed:
            fa_acts.append(act)
            fa_by_age[age] = fa_by_age.get(age, 0) + 1
    print(f"\n跌倒试验总数 {n_fall} (老年 {fall_trial_by_age.get('老年SE', 0)} / "
          f"青年 {fall_trial_by_age.get('青年SA', 0)})")

    # ---------- 1. 误报来源排名 ----------
    print("\n" + "=" * 62)
    print(f"误报来源排名 (阈值{T}, K={K}, 测试集)")
    print("=" * 62)
    print(f"{'活动':6s} {'描述':12s} {'误报事件数':>8s} {'该活动误报率':>12s}")
    rows = []
    for act in sorted(set(fa_acts)):
        n_fa = fa_acts.count(act)
        # 该活动在测试集的总时长(用于折算误报率)
        secs = sum((test_fid == i).sum() * STEP_SEC
                   for i in np.unique(test_fid) if acts[i] == act and test_y[test_fid == i][0] == 0)
        rows.append((act, n_fa, n_fa / (secs / 3600) if secs else 0.0))
    for act, n_fa, rate in sorted(rows, key=lambda r: -r[1]):
        print(f"{act:6s} {ACT_DESC.get(act, '?'):12s} {n_fa:8d} {rate:10.1f}/h")
    print(f"\n误报总数 {len(fa_acts)}, 按年龄段: " +
          " | ".join(f"{k} {v} 个" for k, v in sorted(fa_by_age.items())))

    # ---------- 2. 漏检跌倒类型 ----------
    print("\n" + "=" * 62)
    print(f"漏检跌倒分析 (共 {len(miss_acts)} 次漏检)")
    print("=" * 62)
    for act in sorted(set(miss_acts)):
        print(f"  {act} {ACT_DESC.get(act, '?')}: 漏检 {miss_acts.count(act)} 次")

    # ---------- 3. 柱状图: 误报活动TOP ----------
    if rows:
        rows_sorted = sorted(rows, key=lambda r: -r[1])[:12]
        labels = [f"{a}\n{ACT_DESC.get(a, '')}" for a, _, _ in rows_sorted]
        vals = [n for _, n, _ in rows_sorted]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, vals, color="tab:orange")
        ax.set_ylabel("误报事件数")
        ax.set_title(f"误报来源 TOP{len(rows_sorted)} (测试集, 阈值{T}/K={K})")
        plt.xticks(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_PATH, dpi=150)
        print(f"\n误报来源图已保存至 {FIG_PATH}")


if __name__ == "__main__":
    main()
