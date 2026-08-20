"""
forming.py — 忆阻器渐进成形脚本
==================================
器件栈: Ni / [TiOx 8nm / AlOx 2nm]×3 / Ti

成形策略:
  - 台阶式电压爬升 (0 → V_MAX)，每步驻留后采集电流
  - 检测到电流突变（≥ 0.8 × I_cc）立即回零，避免硬成形损伤
  - 超晶格三层界面可能出现 2–3 个电流台阶，全部记录
  - 支持多档 I_cc 梯队：低档失败后自动升档重试
  - 成形完成后读取 LRS 基线电阻
  - 所有数据保存至 Forming/<条件>_<时间戳>/ 子文件夹

依赖: pip install pyvisa pyvisa-py numpy matplotlib
"""

import os
import sys
import time
import argparse
from datetime import datetime

import matplotlib
import platform
_fonts = {"Windows": "Microsoft YaHei", "Darwin": "PingFang SC", "Linux": "WenQuanYi Micro Hei"}
matplotlib.rcParams["font.family"]        = _fonts.get(platform.system(), "DejaVu Sans")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ══════════════════════════════════════════════════════
#  ★ 可调参数区 ★
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = "USB0::0x05E6::0x2450::04437634::INSTR"

V_START       =  0.0     # 起始电压 (V)
V_MAX         =  4.0     # 电压上限 (V)  — 到达此值仍未成形则中止
V_STEP        =  0.05    # 每步电压增量 (V)
DWELL_TIME    =  0.5     # 每步驻留时间 (s)
NPLC          =  1.0     # 积分时间

# I_cc 梯队：成形失败后依次升档重试
# 格式: [(I_cc_A, 描述), ...]
ICC_LADDER = [
    (50e-6,  "Stage 1: 50 µA  (保守)"),
    (100e-6, "Stage 2: 100 µA"),
    (200e-6, "Stage 3: 200 µA"),
]

# 成形判定：电流达到 I_cc 的此倍数时触发
FORM_THRESHOLD = 0.8

# 台阶检测：dI/dV 超过此值视为电流台阶（超晶格多界面特征）
STEP_DETECT_FACTOR = 5.0   # 相对于前10点平均 dI/dV 的倍数

# 成形后 LRS 基线读取
V_READ        =  0.1     # 读取电压 (V)
N_READ        =  10      # 采样次数
READ_SETTLE   =  5.0     # 成形后等待弛豫时间 (s)
# ══════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Forming")


# ──────────────────────────────────────────────────────
#  连接
# ──────────────────────────────────────────────────────
def connect(address: str):
    try:
        import pyvisa
    except ImportError:
        print("[ERROR] pip install pyvisa pyvisa-py"); sys.exit(1)

    rm = pyvisa.ResourceManager()
    print(f"[连接] {address}")
    try:
        inst = rm.open_resource(address)
    except Exception as e:
        print(f"[ERROR] {e}")
        for r in rm.list_resources(): print(f"  可用: {r}")
        sys.exit(1)

    inst.write_termination = "\n"
    inst.read_termination  = "\n"
    inst.timeout           = 15000

    try:
        inst.clear(); time.sleep(0.3)
    except Exception:
        pass
    inst.write("*CLS"); time.sleep(0.1)

    idn = inst.query("*IDN?").strip()
    print(f"[连接] 成功 -> {idn}\n")
    return inst


# ──────────────────────────────────────────────────────
#  SMU 初始化
# ──────────────────────────────────────────────────────
def init_smu(inst, i_cc: float, nplc: float):
    cmds = [
        "*RST", "*CLS",
        ":SOUR:FUNC VOLT",
        ":SOUR:VOLT:RANG:AUTO 1",
        f":SOUR:VOLT:ILIM {i_cc:.6e}",
        ":SOUR:VOLT:LEV 0",
        ':SENS:FUNC "CURR"',
        ":SENS:CURR:RANG:AUTO 1",
        f":SENS:CURR:NPLC {nplc:.2f}",
        ":SYST:AZER:ONCE",
        ":OUTP OFF",
    ]
    for c in cmds:
        inst.write(c)
    time.sleep(0.3)

    err = inst.query(":SYST:ERR?").strip()
    if not err.startswith("0"):
        print(f"[警告] 初始化: {err}")
    else:
        print(f"[初始化] I_cc = {i_cc*1e6:.0f} µA，完成")


# ──────────────────────────────────────────────────────
#  单步读电流
# ──────────────────────────────────────────────────────
def read_current(inst) -> float:
    try:
        raw   = inst.query(":READ?").strip()
        parts = raw.split(",")
        return float(parts[1]) if len(parts) >= 2 else float(parts[0])
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────
#  台阶检测辅助
# ──────────────────────────────────────────────────────
def detect_step(voltages, currents, factor=STEP_DETECT_FACTOR) -> list:
    """
    从已采集的 (V, I) 数据中检测电流突变台阶。
    返回台阶位置列表: [(V_step, I_step), ...]
    """
    if len(currents) < 12:
        return []

    steps = []
    i_arr = np.array(currents)
    v_arr = np.array(voltages)
    di    = np.abs(np.diff(i_arr))

    # 基准噪声：前10步的平均 dI
    baseline = np.mean(di[:10]) if len(di) >= 10 else np.mean(di)
    if baseline < 1e-12:
        baseline = 1e-12

    threshold = baseline * factor
    in_step = False
    for k in range(len(di)):
        if di[k] > threshold and not in_step:
            steps.append((float(v_arr[k+1]), float(i_arr[k+1])))
            in_step = True
        elif di[k] <= threshold:
            in_step = False

    return steps


# ──────────────────────────────────────────────────────
#  成形主循环（单档 I_cc）
# ──────────────────────────────────────────────────────
def forming_ramp(inst, i_cc: float, stage_label: str,
                 v_start, v_max, v_step, dwell) -> dict:
    """
    台阶式电压爬升，返回成形结果字典。
    result["formed"] = True/False
    """
    v_seq     = np.arange(v_start, v_max + v_step * 0.5, v_step)
    voltages  = []
    currents  = []
    formed    = False
    v_forming = None
    i_forming = None

    print(f"\n{'─'*54}")
    print(f"  {stage_label}")
    print(f"  V: {v_start} → {v_max} V   步长 {v_step} V   驻留 {dwell} s")
    print(f"{'─'*54}")
    print(f"  {'V (V)':>8}  {'I (µA)':>11}  {'I/I_cc':>8}  {'状态':>6}")
    print(f"  {'─'*40}")

    inst.write(":OUTP ON")

    for v in v_seq:
        inst.write(f":SOUR:VOLT:LEV {v:.4f}")
        time.sleep(dwell)

        i = read_current(inst)
        voltages.append(v)
        currents.append(i)

        ratio = abs(i) / i_cc if i_cc > 0 else 0
        status = ""

        # ── 成形判定 ──────────────────────────────────
        if ratio >= FORM_THRESHOLD:
            status    = "★ FORMED"
            formed    = True
            v_forming = v
            i_forming = i
            print(f"  {v:>8.3f}  {i*1e6:>11.4f}  {ratio:>8.3f}  {status}")
            break

        # ── 台阶检测（实时）───────────────────────────
        if len(currents) >= 3:
            prev_di = abs(currents[-1] - currents[-2])
            base_di = (abs(currents[-2] - currents[-3])
                       if len(currents) >= 3 else prev_di)
            base_di = max(base_di, 1e-12)
            if prev_di > base_di * STEP_DETECT_FACTOR and prev_di > 1e-8:
                status = "↑ STEP"

        print(f"  {v:>8.3f}  {i*1e6:>11.4f}  {ratio:>8.3f}  {status}")

    # ── 立即回零 ──────────────────────────────────────
    inst.write(":SOUR:VOLT:LEV 0")
    inst.write(":OUTP OFF")

    if formed:
        print(f"\n  [成形成功] V_forming = {v_forming:.3f} V  "
              f"I_forming = {i_forming*1e6:.2f} µA")
    else:
        print(f"\n  [未成形] 已到达 V_MAX = {v_max} V，电流未达到阈值")

    return {
        "formed":    formed,
        "v_forming": v_forming,
        "i_forming": i_forming,
        "voltages":  voltages,
        "currents":  currents,
        "i_cc":      i_cc,
        "stage":     stage_label,
    }


# ──────────────────────────────────────────────────────
#  成形后 LRS 基线读取
# ──────────────────────────────────────────────────────
def read_lrs(inst, v_read: float, n_read: int,
             settle: float, i_cc: float) -> dict:
    print(f"\n[LRS] 等待弛豫 {settle:.0f} s...")
    time.sleep(settle)

    # 切换到低顺应电流读取模式
    inst.write(f":SOUR:VOLT:ILIM {i_cc:.6e}")
    inst.write(f":SOUR:VOLT:LEV {v_read:.4f}")
    inst.write(":OUTP ON")
    time.sleep(0.3)

    readings = []
    for _ in range(n_read):
        i = read_current(inst)
        if not np.isnan(i) and abs(i) > 1e-12:
            readings.append(abs(v_read / i))
        time.sleep(0.1)

    inst.write(":SOUR:VOLT:LEV 0")
    inst.write(":OUTP OFF")

    if readings:
        r_mean = float(np.mean(readings))
        r_std  = float(np.std(readings))
    else:
        r_mean = float("inf")
        r_std  = 0.0

    status = ("OK ✓" if r_mean < 50e3
              else "偏高 ⚠" if r_mean < 200e3
              else "成形可疑 ✗")
    print(f"[LRS] R_mean = {r_mean/1e3:.2f} kΩ  "
          f"R_std = {r_std/1e3:.2f} kΩ  → {status}")

    return {"r_mean": r_mean, "r_std": r_std,
            "status": status, "v_read": v_read}


# ──────────────────────────────────────────────────────
#  保存 CSV
# ──────────────────────────────────────────────────────
def save_csv(results: list, lrs: dict, path: str):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "index", "voltage_V",
                    "current_A", "current_uA",
                    "resistance_Ohm", "i_cc_A"])
        for res in results:
            for k, (v, i) in enumerate(zip(res["voltages"], res["currents"])):
                r = abs(v / i) if abs(i) > 1e-12 else float("inf")
                w.writerow([
                    res["stage"], k,
                    f"{v:.4f}", f"{i:.6e}", f"{i*1e6:.4f}",
                    f"{r:.2f}" if np.isfinite(r) else "inf",
                    f"{res['i_cc']:.6e}",
                ])

        # LRS 基线附在末尾
        w.writerow([])
        w.writerow(["# LRS baseline",
                    f"R_mean={lrs['r_mean']:.2f} Ohm",
                    f"R_std={lrs['r_std']:.2f} Ohm",
                    f"V_read={lrs['v_read']} V",
                    lrs["status"]])

    print(f"[CSV] 已保存: {path}")


# ──────────────────────────────────────────────────────
#  绘图
# ──────────────────────────────────────────────────────
STAGE_COLORS = ["#2196F3", "#FF9800", "#E91E63", "#4CAF50"]


def plot_forming(results: list, lrs: dict,
                 v_max: float, img_path: str):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        for sp in ax.spines.values(): sp.set_color("#444")

    # ── 左图: I-V 成形曲线 ──────────────────────────
    for k, res in enumerate(results):
        col   = STAGE_COLORS[k % len(STAGE_COLORS)]
        v_arr = np.array(res["voltages"])
        i_arr = np.array(res["currents"]) * 1e6
        i_cc  = res["i_cc"]
        label = res["stage"].split(":")[0]   # "Stage 1" etc.

        ax1.plot(v_arr, i_arr, color=col, lw=1.8,
                 alpha=0.9, label=label, zorder=3)
        ax1.scatter(v_arr, i_arr, color=col, s=12,
                    alpha=0.6, zorder=4)

        # I_cc 参考线（每档颜色对应）
        ax1.axhline(i_cc * 1e6, color=col, lw=0.8,
                    linestyle=":", alpha=0.6)
        ax1.axhline(i_cc * 1e6 * FORM_THRESHOLD,
                    color=col, lw=0.8, linestyle="--", alpha=0.5)

        # 台阶标注
        steps = detect_step(res["voltages"], res["currents"])
        for n_step, (vs, ist) in enumerate(steps):
            ax1.annotate(
                f"Step {n_step+1}\n{vs:.2f} V",
                xy=(vs, ist*1e6),
                xytext=(vs + 0.1, ist*1e6 + i_cc*1e6*0.08),
                color="#FFD54F", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="#FFD54F", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="#1A1D27", ec="#FFD54F", alpha=0.85))

        # 成形点标注
        if res["formed"] and res["v_forming"] is not None:
            vf = res["v_forming"]
            if_ = res["i_forming"] * 1e6
            ax1.scatter([vf], [if_], color="#FF4081", s=120,
                        zorder=7, marker="*",
                        edgecolors="white", linewidths=0.5)
            ax1.annotate(
                f"Formed\n{vf:.2f} V",
                xy=(vf, if_),
                xytext=(vf - 0.6, if_ - i_cc*1e6*0.15),
                color="#FF4081", fontsize=8,
                arrowprops=dict(arrowstyle="->",
                                color="#FF4081", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="#1A1D27", ec="#FF4081", alpha=0.9))

    ax1.axhline(0, color="#555", lw=0.5)
    ax1.axvline(0, color="#555", lw=0.5)
    ax1.set_xlim(-0.1, v_max + 0.2)
    ax1.set_xlabel("Voltage (V)",  color="#CCCCCC", fontsize=10)
    ax1.set_ylabel("Current (µA)", color="#CCCCCC", fontsize=10)
    ax1.set_title("Forming I–V  (台阶爬升)",
                  color="#EEEEEE", fontsize=11, pad=8)
    ax1.legend(facecolor="#1A1D27", edgecolor="#444",
               labelcolor="#CCCCCC", fontsize=8.5, loc="upper left")
    ax1.grid(True, color="#2A2D37", lw=0.5, zorder=0)

    # 图例补充说明
    patch_icc  = mpatches.Patch(color="gray", linestyle=":",
                                 label=f"I_cc (per stage)")
    patch_thr  = mpatches.Patch(color="gray", linestyle="--",
                                 label=f"0.8 × I_cc (形成阈值)")
    ax1.legend(handles=ax1.get_legend_handles_labels()[0]
               + [patch_icc, patch_thr],
               labels=ax1.get_legend_handles_labels()[1]
               + ["I_cc (per stage)", "0.8×I_cc threshold"],
               facecolor="#1A1D27", edgecolor="#444",
               labelcolor="#CCCCCC", fontsize=7.5, loc="upper left")

    # ── 右图: R-V（成形过程中的阻态演化）──────────
    for k, res in enumerate(results):
        col   = STAGE_COLORS[k % len(STAGE_COLORS)]
        v_arr = np.array(res["voltages"])
        i_arr = np.array(res["currents"])
        safe  = np.abs(i_arr) > 1e-12
        r_arr = np.where(safe, np.abs(v_arr / i_arr), np.nan) / 1e3

        # 只画 V > 0.1V 的点（避免过零点奇点）
        mask = v_arr > 0.1
        if mask.sum() > 1:
            label = res["stage"].split(":")[0]
            ax2.scatter(v_arr[mask], r_arr[mask],
                        c=col, s=15, alpha=0.75,
                        zorder=3, label=label)
            ax2.plot(v_arr[mask], r_arr[mask],
                     color=col, lw=1.2, alpha=0.6, zorder=2)

    # LRS 基线标注
    if lrs and np.isfinite(lrs["r_mean"]):
        r_lrs_k = lrs["r_mean"] / 1e3
        ax2.axhline(r_lrs_k, color="#69F0AE", lw=1.5,
                    linestyle="--", alpha=0.9,
                    label=f"LRS baseline ≈ {r_lrs_k:.1f} kΩ")
        ax2.text(0.15, r_lrs_k * 1.15,
                 f"R_LRS = {r_lrs_k:.2f} kΩ\n"
                 f"(读于 {lrs['v_read']} V，弛豫后)",
                 color="#69F0AE", fontsize=8)

    ax2.set_yscale("log")
    ax2.set_xlim(-0.05, v_max + 0.2)
    ax2.set_xlabel("Voltage (V)",    color="#CCCCCC", fontsize=10)
    ax2.set_ylabel("Resistance (kΩ)", color="#CCCCCC", fontsize=10)
    ax2.set_title("Resistance vs. Forming Voltage",
                  color="#EEEEEE", fontsize=11, pad=8)
    ax2.legend(facecolor="#1A1D27", edgecolor="#444",
               labelcolor="#CCCCCC", fontsize=8.5)
    ax2.grid(True, color="#2A2D37", lw=0.5, which="both", zorder=0)

    # 总标题
    formed_any = any(r["formed"] for r in results)
    v_f_str = "N/A"
    for res in results:
        if res["formed"] and res["v_forming"]:
            v_f_str = f"{res['v_forming']:.2f} V  ({res['stage'].split(':')[0]})"
            break

    r_lrs_str = (f"{lrs['r_mean']/1e3:.2f} kΩ"
                 if lrs and np.isfinite(lrs["r_mean"]) else "N/A")

    status_str = "✓ 成形成功" if formed_any else "✗ 未成形"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"Ni/TiOx-AlOx/Ti   Soft Forming   {status_str}   "
        f"V_form={v_f_str}   R_LRS={r_lrs_str}   {ts}",
        color="#DDDDDD", fontsize=9, y=1.01)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] 已保存: {img_path}")
    plt.show()


# ──────────────────────────────────────────────────────
#  主程序
# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="忆阻器渐进成形")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    parser.add_argument("--dev",     default="DEV",
                        help="器件标签，用于文件夹命名（如 DEV-01）")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 文件夹命名：器件标签 + 成形参数 + 时间戳
    run_name = (
        f"{args.dev}_"
        f"Vmax{V_MAX}_"
        f"step{int(V_STEP*1000)}mV_"
        f"dwell{int(DWELL_TIME*1000)}ms_"
        f"{ts_str}"
    )
    RUN_DIR = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(RUN_DIR, exist_ok=True)

    csv_path = os.path.join(RUN_DIR, f"forming_{ts_str}.csv")
    img_path = os.path.join(RUN_DIR, f"forming_{ts_str}.png")

    # 参数确认
    print("=" * 54)
    print("  渐进成形参数")
    print("=" * 54)
    print(f"  V_MAX       = {V_MAX} V")
    print(f"  V_STEP      = {V_STEP*1000:.0f} mV")
    print(f"  DWELL_TIME  = {DWELL_TIME*1000:.0f} ms / 步")
    print(f"  NPLC        = {NPLC}")
    print(f"  FORM_THR    = {FORM_THRESHOLD} × I_cc")
    print(f"  I_cc 梯队:")
    for icc, label in ICC_LADDER:
        print(f"    {icc*1e6:>7.0f} µA  — {label}")
    n_steps = int((V_MAX - V_START) / V_STEP) + 1
    est_s   = n_steps * DWELL_TIME * len(ICC_LADDER)
    print(f"  最长耗时估计 ≈ {est_s:.0f} s（全部未成形情况）")
    print(f"  器件标签    = {args.dev}")
    print(f"  输出目录    = {RUN_DIR}")
    print("=" * 54)

    ans = input("\n确认开始成形? (Enter / q 退出): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)

    results = []
    formed  = False

    # ── 逐档 I_cc 尝试 ───────────────────────────────
    for stage_idx, (i_cc, label) in enumerate(ICC_LADDER):
        init_smu(inst, i_cc, NPLC)

        res = forming_ramp(
            inst, i_cc, label,
            V_START, V_MAX, V_STEP, DWELL_TIME
        )
        results.append(res)

        if res["formed"]:
            formed = True
            break

        if stage_idx < len(ICC_LADDER) - 1:
            print(f"\n  [升档] 尝试下一档 I_cc...")
            time.sleep(1.0)

    # ── 成形后基线读取 ───────────────────────────────
    lrs = {"r_mean": float("inf"), "r_std": 0.0,
           "status": "N/A", "v_read": V_READ}

    if formed:
        # 切回低 I_cc 保护读取
        read_i_cc = min(i_cc, 200e-6)
        init_smu(inst, read_i_cc, NPLC)
        lrs = read_lrs(inst, V_READ, N_READ, READ_SETTLE, read_i_cc)
    else:
        print("\n[结果] 所有档位均未成功成形，跳过 LRS 读取")

    inst.close()

    # ── 汇总打印 ─────────────────────────────────────
    print("\n" + "=" * 54)
    print("  成形结果汇总")
    print("=" * 54)
    print(f"  成形状态  : {'成功 ✓' if formed else '失败 ✗'}")
    for res in results:
        tag = "★" if res["formed"] else " "
        vf  = f"{res['v_forming']:.3f} V" if res["v_forming"] else "—"
        print(f"  {tag} {res['stage']:<28} V_form = {vf}")
    if formed:
        print(f"  R_LRS     : {lrs['r_mean']/1e3:.2f} kΩ  {lrs['status']}")

        steps_all = []
        for res in results:
            if res["formed"]:
                steps_all = detect_step(res["voltages"], res["currents"])
                break
        if steps_all:
            print(f"  电流台阶  : {len(steps_all)} 个（超晶格界面特征）")
            for n, (vs, _) in enumerate(steps_all):
                print(f"    Step {n+1}: {vs:.3f} V")
    print("=" * 54)

    # ── 保存 & 绘图 ──────────────────────────────────
    save_csv(results, lrs, csv_path)
    plot_forming(results, lrs, V_MAX, img_path)


if __name__ == "__main__":
    main()