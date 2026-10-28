# -*- coding: utf-8 -*-
"""评估脚本（四部分）:
Part1 默认阈值0.5的整体指标 + 混淆矩阵图
Part2 阈值扫描(在验证集上) -> 召回率/特异度随阈值变化曲线 + 推荐工作点
Part3 分组评估: 青年(SA) vs 老年(SE)受试者分别汇报
Part4 事件级评估: 按试验(文件)聚合 -> 事件召回率 + 每小时误报数
"""
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能画图
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from dataset import get_datasets, load_npz, split_by_subject
from model import FallNet

OUT_DIR = r"c:\Users\ASUS\Documents\trae_projects\fall detection\outputs"
MODEL_PATH = OUT_DIR + r"\best_model.pt"
FIG_PATH = OUT_DIR + r"\confusion_matrix.png"
CURVE_PATH = OUT_DIR + r"\threshold_curve.png"
CALIB_PATH = OUT_DIR + r"\calibration_curve.png"  # 校准曲线(可靠性图)输出路径

STEP_SEC = 0.5   # 滑窗步长 0.5s: 用于估算时长/每小时误报
SPEC_MIN = 0.80  # 推荐阈值约束: 验证集特异度不低于 80%

# 解决 matplotlib 中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


@torch.no_grad()
def predict_probs(model, loader, device):
    """返回每个窗口属于正类(跌倒)的概率"""
    probs = []
    model.eval()
    for xb, _ in loader:
        p = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
    return np.array(probs)


def metrics_at(y_true, y_pred):
    """由混淆矩阵展开计算各项指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(acc=(tp + tn) / len(y_true), recall=tp / (tp + fn),
                spec=tn / (tn + fp), prec=tp / (tp + fp) if tp + fp else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)


def print_metrics(tag, y_true, y_pred):
    m = metrics_at(y_true, y_pred)
    print(f"  [{tag:6s}] Acc {m['acc']:.4f} | Recall {m['recall']:.4f} | "
          f"Spec {m['spec']:.4f} | Prec {m['prec']:.4f} | F1 {m['f1']:.4f}")
    return m


def reliability_diagram(y_true, probs, n_bins=10):
    """计算校准度相关指标: ECE(期望校准误差) + 各分箱的准确率/置信度

    原理: 将预测概率分成 n_bins 个等宽区间, 对每个区间统计
      - 置信度 bin_conf = 该区间所有样本预测概率的均值
      - 准确率 bin_acc  = 该区间真实正例比例
    理想校准下 bin_acc ≈ bin_conf; ECE 越小说明概率越可靠.

    参数:
        y_true: 真实标签 {0,1}
        probs : 模型预测的正类概率 [0,1]
        n_bins: 分箱数
    返回:
        bin_accs, bin_confs, ece (float)
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs = [], []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(probs[mask].mean())
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        # ECE = Σ |bin_acc - bin_conf| * (bin_count / total)
        ece += abs(bin_acc - bin_conf) * mask.sum() / len(y_true)
    return bin_accs, bin_confs, ece


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    _, val_set, test_set = get_datasets()
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    # 直接从 npz 取原始标签/元信息（顺序与 split 一致）
    x, y, subj, fid = load_npz()
    _, va, te = split_by_subject(subj)
    val_y, test_y = y[va], y[te]
    test_subj, test_fid = subj[te], fid[te]

    val_p = predict_probs(model, val_loader, device)
    test_p = predict_probs(model, test_loader, device)

    # ================= Part 1: 默认阈值 0.5 =================
    pred05 = (test_p >= 0.5).astype(int)
    print("\n" + "=" * 60)
    print("Part1 测试集整体指标 (默认阈值 0.5)")
    print("=" * 60)
    print_metrics("全部", test_y, pred05)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(test_y, pred05), display_labels=["日常活动", "跌倒"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("测试集混淆矩阵 (阈值0.5)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"混淆矩阵图已保存至 {FIG_PATH}")

    # ================= Part 1.5: 校准曲线 + ECE =================
    print("\n" + "=" * 60)
    print("Part1.5 校准度评估 (ECE = 期望校准误差, 越小越可靠)")
    print("=" * 60)
    bin_accs, bin_confs, ece = reliability_diagram(test_y, test_p, n_bins=10)
    print(f"  测试集 ECE = {ece:.4f}  (ECE 越接近 0 表示概率越可靠)")

    # 画可靠性图: 理想校准是 y=x 对角线, 柱顶偏离对角线越远说明越不准
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_positions = np.arange(1, len(bin_accs) + 1) / len(bin_accs)
    ax.bar(bar_positions, bin_accs, width=1.0 / len(bin_accs) * 0.8,
           color="tab:blue", alpha=0.7, label="实际准确率")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="理想校准 (y=x)")
    ax.scatter(bin_confs, bin_accs, color="tab:red", zorder=5,
               label=f"分箱准确率 (ECE={ece:.3f})")
    ax.set_xlabel("平均预测概率 (置信度)")
    ax.set_ylabel("实际准确率")
    ax.set_title(f"校准曲线 (可靠性图) — ECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CALIB_PATH, dpi=150)
    print(f"校准曲线图已保存至 {CALIB_PATH}")

    # ================= Part 2: 阈值扫描(验证集) =================
    print("\n" + "=" * 60)
    print(f"Part2 阈值扫描 (验证集, 约束特异度 >= {SPEC_MIN:.0%} 下召回率最大)")
    print("=" * 60)
    thresholds = np.arange(0.05, 0.96, 0.05)
    recs, specs = [], []
    for t in thresholds:
        m = metrics_at(val_y, (val_p >= t).astype(int))
        recs.append(m["recall"])
        specs.append(m["spec"])
    recs, specs = np.array(recs), np.array(specs)
    ok = np.where(specs >= SPEC_MIN)[0]
    if len(ok):
        best_t = float(thresholds[ok[np.argmax(recs[ok])]])
    else:
        best_t = 0.5
        print("警告: 无满足特异度约束的阈值, 回退 0.5")
    print(f"推荐阈值: {best_t:.2f}  (验证集 Recall {recs[ok[np.argmax(recs[ok])]] if len(ok) else 0:.4f})")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(thresholds, recs, "o-", label="验证集召回率", color="tab:red")
    ax.plot(thresholds, specs, "s-", label="验证集特异度", color="tab:blue")
    ax.axvline(0.5, ls="--", c="gray", lw=1, label="默认阈值 0.5")
    ax.axvline(best_t, ls="--", c="tab:green", lw=1.5, label=f"推荐阈值 {best_t:.2f}")
    ax.set_xlabel("决策阈值 (跌倒概率 >= 阈值 判为跌倒)")
    ax.set_ylabel("指标值")
    ax.set_title("阈值扫描: 召回率-特异度权衡曲线")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=150)
    print(f"阈值曲线图已保存至 {CURVE_PATH}")

    # ================= Part 3: 分组评估(青年/老年) =================
    print("\n" + "=" * 60)
    print("Part3 分组评估 (测试集按受试者年龄段)")
    print("=" * 60)
    sa_mask = np.char.startswith(test_subj, "SA")   # 青年受试者
    se_mask = np.char.startswith(test_subj, "SE")   # 老年受试者
    for tag, t in [("阈值0.5", 0.5), (f"阈值{best_t:.2f}", best_t)]:
        pred = (test_p >= t).astype(int)
        print(f"--- 测试集 @ {tag} ---")
        print_metrics("全部", test_y, pred)
        print_metrics("青年SA", test_y[sa_mask], pred[sa_mask])
        print_metrics("老年SE", test_y[se_mask], pred[se_mask])

    # ================= Part 4: 事件级评估 =================
    print("\n" + "=" * 60)
    print("Part4 事件级评估 (按试验聚合: 一次跌倒只要有>=1个窗口报警即算检出)")
    print("=" * 60)
    for tag, t in [("阈值0.5", 0.5), (f"阈值{best_t:.2f}", best_t)]:
        pred = (test_p >= t).astype(int)
        fall_total = fall_hits = fa_events = 0
        adl_seconds = 0.0
        for f in np.unique(test_fid):
            mask = test_fid == f
            if test_y[mask][0] == 1:            # 跌倒试验: 任一窗口报警即检出
                fall_total += 1
                fall_hits += int(pred[mask].any())
            else:                                # 日常试验: 任一窗口报警即一次误报事件
                fa_events += int(pred[mask].any())
                adl_seconds += mask.sum() * STEP_SEC
        fa_per_hour = fa_events / (adl_seconds / 3600) if adl_seconds else 0.0
        print(f"  @ {tag:10s} 跌倒事件召回 {fall_hits}/{fall_total} ({fall_hits/max(fall_total,1):.2%})"
              f" | 误报事件 {fa_events} 个 | 每小时误报 {fa_per_hour:.1f} 次")


if __name__ == "__main__":
    main()
