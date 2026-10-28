# -*- coding: utf-8 -*-
"""三级流水线 Demo: 各选一个跌倒试验与一个躺下类日常试验, 画出流水线每级的判定时间线
流程: ①窗口分类(概率>=0.6) -> ②时序确认(连续K=6窗均超阈值) -> ③静止验真(报警后3s运动分数<0.02)
概率复用 probs_cache.npz(与 best_model.pt 同一模型的推理缓存), 运动分数按需现算
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_npz, split_by_subject
from temporal_confirm import trial_alarm, PROBS_PATH
from stillness_verify import movement_scores, alarm_clusters

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs"
FIG_PATH = OUT_DIR + r"\demo_pipeline.png"
DATA_DIR = r"C:\Users\ASUS\Desktop\fall database\archive (1)\SisFall_dataset"
DATA_PATTERN = re.compile(r"^[DF]\d+_[A-Z]+\d+_R\d+\.txt$", re.IGNORECASE)

T, K = 0.6, 6            # ①②级配置(验证集扫参选定)
D_SEC, S_TH = 3.0, 0.02  # ③级配置(验证集扫参选定)
STEP_SEC, FS = 0.5, 200  # 窗口步长(秒) / 采样率(Hz)

# SisFall 官方活动对照表(中译), 与 fa_analysis.py 保持一致
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


def trial_signal(xw):
    """由重叠窗口 [n,200,6] 还原连续信号: 首窗全取200点, 后续窗各取末尾100点(相邻窗重叠100点)"""
    tail = xw[1:, -100:].reshape(-1, xw.shape[2])   # [n-1,100,6] -> [(n-1)*100,6]
    return np.vstack([xw[0], tail])


def pipeline_decision(p, m):
    """对单个试验执行三级流水线
    返回 (报警簇列表, 被确认的簇下标或None, 每个簇报警后D秒运动分数均值(None=报警持续到记录末尾))"""
    clusters = alarm_clusters(trial_alarm(p, T, K))
    w_follow = int(D_SEC / STEP_SEC)
    verdicts = []
    for st, en in clusters:
        lo, hi = en, min(en + w_follow, len(m))
        verdicts.append(None if hi <= lo else m[lo:hi].mean())
    confirmed = next((i for i, v in enumerate(verdicts) if v is None or v < S_TH), None)
    return clusters, confirmed, verdicts


def pick_trials(test_p, test_y, test_fid, x_test, m_test, files):
    """挑演示试验: 跌倒=被流水线检出且冲击峰值最大; 日常=有报警但全被静止验真撤销, 优先躺下翻身类"""
    fall_c, adl_c = [], []
    for f in np.unique(test_fid):
        mask = test_fid == f
        clusters, confirmed, _ = pipeline_decision(test_p[mask], m_test[mask])
        name = os.path.basename(files[f])[:-4]
        act, s = name.split("_")[0], name.split("_")[1]
        if test_y[mask][0] == 1 and confirmed is not None:
            peak = np.linalg.norm(x_test[mask][:, :, 0:3], axis=2).max()
            fall_c.append((peak, f, name, act, s))
        elif test_y[mask][0] == 0 and len(clusters) and confirmed is None:
            adl_c.append((act, len(clusters), f, name, s))

    fall_c.sort(key=lambda c: -c[0])                       # 冲击峰值最大 = 最典型跌倒
    pref = {"D14": 0, "D12": 1, "D16": 2}                  # 优先展示误报最集中的躺下/弯腰类
    adl_c.sort(key=lambda c: (pref.get(c[0], 9), -c[1]))
    if not fall_c or not adl_c:
        raise RuntimeError(f"候选不足: 检出跌倒 {len(fall_c)} 个, 被撤销ADL {len(adl_c)} 个")
    # 统一返回 (fid, 文件名, 活动代码, 受试者)
    return fall_c[0][1:], (adl_c[0][2], adl_c[0][3], adl_c[0][0], adl_c[0][4])


def plot_trial(axs, sig, p, m, clusters, confirmed, verdicts, title):
    """画一个试验的三行时间线: 原始|acc| / 跌倒概率 / 运动分数"""
    t_sig = np.arange(len(sig)) / FS
    t_win = np.arange(len(p)) * STEP_SEC + 0.5     # 窗口中心时刻
    for i, (st, en) in enumerate(clusters):
        t0, t1 = st * STEP_SEC, (en - 1) * STEP_SEC + 1.0
        color = "red" if i == confirmed else "orange"
        for ax in axs:
            ax.axvspan(t0, t1, color=color, alpha=0.15)

    # 行1: 原始加速度模长
    axs[0].plot(t_sig, np.linalg.norm(sig[:, 0:3], axis=1), lw=0.6, color="tab:blue")
    axs[0].set_ylabel("|acc| (g)")
    axs[0].set_title(title, fontsize=11)

    # 行2: 窗口级跌倒概率 + 分类阈值
    axs[1].plot(t_win, p, drawstyle="steps-mid", lw=1.2, color="tab:purple", label="窗口跌倒概率")
    axs[1].axhline(T, color="red", ls="--", lw=1, label=f"分类阈值 {T}")
    axs[1].set_ylabel("跌倒概率 p")
    axs[1].set_ylim(-0.03, 1.08)
    axs[1].legend(loc="upper right", fontsize=8)

    # 行3: 运动分数 + 静止阈值 + 报警后观察区
    axs[2].plot(t_win, m, lw=1.0, color="tab:green", label="运动分数")
    axs[2].axhline(S_TH, color="red", ls="--", lw=1, label=f"静止阈值 {S_TH}")
    for i, (st, en) in enumerate(clusters):
        t0 = en * STEP_SEC                          # 观察区从报警簇结束开始
        axs[2].axvspan(t0, t0 + D_SEC, color="gray", alpha=0.2)
        if verdicts[i] is not None:
            axs[2].plot(t0 + D_SEC / 2, verdicts[i], "kv", ms=6)
    axs[2].set_ylabel("运动分数")
    axs[2].set_xlabel("时间 (秒)")
    axs[2].legend(loc="upper right", fontsize=8)

    # 判定结论文本框
    if confirmed is not None:
        v = verdicts[confirmed]
        note = (f"判定: [确认跌倒] 第{confirmed+1}个报警簇通过静止验真\n"
                + (f"报警后{D_SEC:.0f}s运动分数均值 {v:.3f} < {S_TH} (跌倒后持续静止)"
                   if v is not None else "报警持续到记录末尾, 保守确认"))
    else:
        v = np.mean([x for x in verdicts if x is not None])
        note = (f"判定: [撤销警报] {len(clusters)}个报警簇均未通过静止验真\n"
                f"报警后{D_SEC:.0f}s运动分数均值 {v:.3f} >= {S_TH} (人已再次活动, 非跌倒)")
    axs[2].text(0.02, 0.97, note, transform=axs[2].transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

    axs[0].set_xlim(0, max(t_sig[-1], t_win[-1]) + 0.5)


def main():
    cache = np.load(PROBS_PATH)
    x, y, subj, fid = load_npz()
    _, _, te = split_by_subject(subj)
    test_p, test_y, test_fid = cache["test_p"], y[te], fid[te]
    x_test = x[te]
    del x
    print("计算测试集运动分数...")
    m_test = movement_scores(x_test)

    files = rebuild_files()
    print(f"文件列表重建: {len(files)} 个 (最大 fid={test_fid.max()}, 一致性校验)")

    (ff, fname_f, act_f, subj_f), (fa, fname_a, act_a, subj_a) = \
        pick_trials(test_p, test_y, test_fid, x_test, m_test, files)
    print(f"\n跌倒演示试验: {fname_f} ({ACT_DESC[act_f]}, 受试者 {subj_f})")
    print(f"日常演示试验: {fname_a} ({ACT_DESC[act_a]}, 受试者 {subj_a})")

    # 组装两列绘图数据
    panels = []
    for f, fname, act, subj_, tag in [
        (ff, fname_f, act_f, subj_f, "跌倒"),
        (fa, fname_a, act_a, subj_a, "日常"),
    ]:
        mask = test_fid == f
        clusters, confirmed, verdicts = pipeline_decision(test_p[mask], m_test[mask])
        panels.append((trial_signal(x_test[mask]), test_p[mask], m_test[mask],
                       clusters, confirmed, verdicts,
                       f"{tag}试验 {fname}\n{ACT_DESC[act]} (受试者 {subj_})"))
        print(f"  [{tag}] 报警簇 {len(clusters)} 个, 判定: "
              + ("确认" if confirmed is not None else "撤销"))

    # ---------- 绘图: 2列 x 3行 ----------
    fig, axs = plt.subplots(3, 2, figsize=(14, 8.5))
    for col, args in enumerate(panels):
        plot_trial(axs[:, col], *args)
    fig.suptitle(f"三级流水线判定示例: 窗口分类(阈值{T}) → 时序确认(K={K}) → 静止验真(D={D_SEC:.0f}s, s={S_TH})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\n演示图已保存至 {FIG_PATH}")


if __name__ == "__main__":
    main()
