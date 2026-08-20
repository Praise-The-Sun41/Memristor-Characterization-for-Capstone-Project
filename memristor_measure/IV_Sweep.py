"""
iv_sweep.py  v4 — 非对称限流版 (SET限流 / RESET放开)
=====================================================
在 v3 逐点步进基础上，支持 SET 方向和 RESET 方向使用不同的
顺应电流(I_cc):
  - SET 方向:  限流(控制细丝粗细，防止过生长)
  - RESET 方向: 放开(给足电流拉断细丝)

诊断目的:
  上一轮发现器件有 SET (~1.8V) 但 RESET 失败。
  对称限流(100µA)可能把 RESET 也压制了。
  本版让 RESET 方向用更大电流，区分:
    放开后出现 RESET → 之前是测量问题(限流压制)
    放开后仍无 RESET → Ti 电极吸氧导致细丝断不开(器件问题)

器件栈: Ni / AlOx / TiOx / Ti
依赖: pip install pyvisa pyvisa-py numpy matplotlib
"""

import os, sys, time, argparse
from datetime import datetime

import matplotlib
import platform
_fonts = {"Windows": "Microsoft YaHei", "Darwin": "PingFang SC", "Linux": "WenQuanYi Micro Hei"}
matplotlib.rcParams["font.family"]        = _fonts.get(platform.system(), "DejaVu Sans")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# ══════════════════════════════════════════════════════
#  ★ 可调参数区 ★
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = "USB0::0x05E6::0x2450::04437634::INSTR"

V_MAX    =  2.0    # 正向扫描上限 (V)
V_MIN    = -2.0    # 负向扫描下限 (V)
N_POINTS =  100    # 每 1/4 段采样点数（完整周期 = N_POINTS × 4）
N_CYCLES =  2     # 扫描周期数
NPLC     =  0.5    # 积分时间

# ── 非对称限流 ★核心改动★ ─────────────────────────
#   SET 方向限流(细)、RESET 方向放开(粗)
#   SET_POLARITY 指定哪个极性是 SET 方向:
#     "positive" = 正压 SET / 负压 RESET (默认，匹配 +1.8V SET 的图)
#     "negative" = 负压 SET / 正压 RESET (若极性相反则改这里)
SET_POLARITY = "positive"

I_CC_SET   = 1000e-6   # SET 方向顺应电流 (A)，限流控制细丝粗细
I_CC_RESET = 5000e-6    # RESET 方向顺应电流 (A)，5000e-6 = 5 mA
                      #   建议从 10mA 起步; 若器件电流本就大可调到 50mA
                      #   ★安全提醒: RESET电流过大有再次过生长/损伤风险,
                      #             从小往大试 (1mA→10mA→50mA)
# ══════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IV_sweep")
CYCLE_PALETTE = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]


# ──────────────────────────────────────────────────────
#  根据电压符号 + 极性设置，返回该步应使用的 I_cc
# ──────────────────────────────────────────────────────
def icc_for_voltage(v: float) -> float:
    """
    正压 SET 模式: V>=0 用 I_CC_SET, V<0 用 I_CC_RESET
    负压 SET 模式: 反之
    V=0 默认用 SET 限流(更保守)
    """
    if SET_POLARITY == "positive":
        return I_CC_SET if v >= 0 else I_CC_RESET
    else:  # negative
        return I_CC_SET if v <= 0 else I_CC_RESET


# ──────────────────────────────────────────────────────
def connect(address):
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
    inst.timeout           = 10000

    try:
        inst.clear(); time.sleep(0.3)
    except Exception:
        pass
    inst.write("*CLS"); time.sleep(0.1)

    idn = inst.query("*IDN?").strip()
    print(f"[连接] 成功 -> {idn}\n")
    return inst


def set_current_limit(inst, requested_icc: float) -> float:
    """Set source current compliance and return the instrument-reported value."""
    candidates = [
        (":SOUR:VOLT:ILIM:LEV", ":SOUR:VOLT:ILIM:LEV?"),
        (":SOUR:VOLT:ILIM", ":SOUR:VOLT:ILIM?"),
    ]

    errors = []
    actual = requested_icc
    accepted = False

    for set_cmd, query_cmd in candidates:
        inst.write(f"{set_cmd} {requested_icc:.6e}")
        time.sleep(0.02)

        err = inst.query(":SYST:ERR?").strip()
        if not err.startswith("0"):
            errors.append(f"{set_cmd}: {err}")
            continue

        try:
            actual = float(inst.query(query_cmd).strip())
            q_err = inst.query(":SYST:ERR?").strip()
            if not q_err.startswith("0"):
                errors.append(f"{query_cmd}: {q_err}")
                actual = requested_icc
            else:
                accepted = True
                break
        except Exception as e:
            errors.append(f"{query_cmd}: {e}")
            actual = requested_icc
            accepted = True
            break

    if not accepted:
        raise RuntimeError(
            f"Instrument rejected Icc={requested_icc:.6e} A "
            f"({requested_icc * 1e3:.3f} mA). Tried: {' | '.join(errors)}"
        )

    if abs(actual - requested_icc) > max(abs(requested_icc) * 0.01, 1e-9):
        print(
            f"[Warning] Requested Icc={requested_icc:.6e} A, "
            f"instrument reports {actual:.6e} A"
        )

    return actual


def init_smu(inst, i_cc_init, nplc):
    cmds = [
        "*RST", "*CLS",
        ":SOUR:FUNC VOLT",
        ":SOUR:VOLT:RANG:AUTO 1",
        ":SOUR:VOLT:LEV 0",
        ':SENS:FUNC "CURR"',
        ":SENS:CURR:RANG:AUTO 1",
        f":SENS:CURR:NPLC {nplc:.2f}",
        ":SYST:AZER:ONCE",
        ":OUTP OFF",
    ]
    for c in cmds:
        inst.write(c)
    actual_icc = set_current_limit(inst, i_cc_init)
    time.sleep(0.3)

    err = inst.query(":SYST:ERR?").strip()
    if not err.startswith("0"):
        print(f"[警告] 初始化: {err}")
    else:
        print("[初始化] 完成，无错误")


def build_cycle(v_max, v_min, n):
    s1 = np.linspace(0,     v_max, n, endpoint=False)
    s2 = np.linspace(v_max, 0,     n, endpoint=False)
    s3 = np.linspace(0,     v_min, n, endpoint=False)
    s4 = np.linspace(v_min, 0,     n, endpoint=True)
    return np.concatenate([s1, s2, s3, s4])


def compute_resistance_with_icc_hold(voltages, currents, icc_used,
                                     trigger_ratio=0.985):
    """
    Compute resistance while holding it constant in current-compliance regions.

    When the source reaches Icc, V/I creates an artificial resistance trend
    because current is clipped by the compliance limit. Hold the boundary
    resistance until current drops back below the compliance threshold.
    """
    voltages = np.asarray(voltages, dtype=float)
    currents = np.asarray(currents, dtype=float)
    icc_used = np.asarray(icc_used, dtype=float)
    resistance = np.full(len(voltages), np.nan, dtype=float)

    held_r = None
    in_compliance = False

    for k, (v, i, icc) in enumerate(zip(voltages, currents, icc_used)):
        if not (np.isfinite(v) and np.isfinite(i)) or abs(i) <= 1e-12:
            resistance[k] = held_r if in_compliance and held_r is not None else np.nan
            continue

        raw_r = abs(v / i)
        at_compliance = np.isfinite(icc) and icc > 0 and abs(i) >= abs(icc) * trigger_ratio

        if at_compliance:
            if not in_compliance:
                held_r = resistance[k - 1] if k > 0 and np.isfinite(resistance[k - 1]) else raw_r
                in_compliance = True
            resistance[k] = held_r
        else:
            in_compliance = False
            held_r = None
            resistance[k] = raw_r

    return resistance


# ──────────────────────────────────────────────────────
def run_sweep(inst, v_max, v_min, n_points, n_cycles, nplc):
    """
    Python逐点步进 + 动态非对称限流:
      每步根据电压符号切换 I_cc (SET方向限流 / RESET方向放开)
    """
    one_cycle = build_cycle(v_max, v_min, n_points)
    cycle_len = len(one_cycle)
    total     = cycle_len * n_cycles

    step_delay = max(0.02, nplc / 60 * 1.5 + 0.02)
    est_s = total * step_delay
    print(f"[扫描] {n_cycles} 周期 × {cycle_len} 点 = {total} 点")
    print(f"[扫描] 步进延迟 {step_delay*1000:.0f} ms/点，预计 {est_s:.0f} s")
    print(f"[扫描] 非对称限流: SET={I_CC_SET*1e6:.0f}µA  "
          f"RESET={I_CC_RESET*1e3:.1f}mA  (SET极性={SET_POLARITY})")

    voltages   = np.zeros(total)
    currents   = np.zeros(total)
    timestamps = np.zeros(total)
    icc_used   = np.zeros(total)   # 记录每点实际用的限流

    inst.write(":OUTP ON")
    t_start = time.time()

    # 跟踪当前 I_cc，仅在切换时才发命令(减少通信开销)
    cur_icc = None

    print(f"\n  {'进度':>6}  {'V (V)':>9}  {'I (µA)':>11}  {'R (kΩ)':>10}  {'I_cc':>8}")
    print("  " + "─" * 56)

    for idx in range(total):
        v_set = one_cycle[idx % cycle_len]

        # ── 动态切换限流 ──────────────────────────────
        want_icc = icc_for_voltage(v_set)
        if cur_icc is None or abs(want_icc - cur_icc) > 1e-12:
            cur_icc = set_current_limit(inst, want_icc)

        inst.write(f":SOUR:VOLT:LEV {v_set:.6f}")
        time.sleep(step_delay)

        try:
            raw   = inst.query(":READ?").strip()
            parts = raw.split(",")
            i     = float(parts[1]) if len(parts) >= 2 else float(parts[0])
        except Exception as e:
            print(f"  [读数警告 #{idx}] {e}")
            i = float("nan")

        voltages[idx]   = v_set
        currents[idx]   = i
        timestamps[idx] = time.time() - t_start
        icc_used[idx]   = cur_icc

        report_every = max(1, cycle_len // 4)
        if idx % report_every == 0 or idx == total - 1:
            r_str = f"{abs(v_set/i)/1e3:.2f}" if abs(i) > 1e-11 else "  inf"
            pct   = (idx + 1) / total * 100
            icc_s = (f"{cur_icc*1e3:.1f}mA" if cur_icc >= 1e-3
                     else f"{cur_icc*1e6:.0f}µA")
            print(f"  {pct:5.1f}%  {v_set:>9.4f}  {i*1e6:>11.4f}  "
                  f"{r_str:>10}  {icc_s:>8}")

    inst.write(":OUTP OFF")
    inst.write(":SOUR:VOLT:LEV 0")

    elapsed = time.time() - t_start
    print(f"\n[扫描] 完成，实际耗时 {elapsed:.1f} s")
    return voltages, currents, timestamps, cycle_len, icc_used


# ──────────────────────────────────────────────────────
def save_csv(voltages, currents, timestamps, resistance, icc_used,
             cycle_len, path, output_dir=None, ts_str=None):
    import csv
    n = len(voltages)
    n_cycles = n // cycle_len

    header = ["index", "cycle", "index_in_cycle", "time_s",
              "voltage_V", "current_A", "current_uA",
              "resistance_Ohm", "icc_A"]

    def row(k):
        r = resistance[k]
        return [k, k // cycle_len, k % cycle_len,
                f"{timestamps[k]:.4f}",
                f"{voltages[k]:.6f}",
                f"{currents[k]:.6e}",
                f"{currents[k]*1e6:.4f}",
                f"{r:.2f}" if np.isfinite(r) else "inf",
                f"{icc_used[k]:.6e}"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(n):
            w.writerow(row(k))
    print(f"[CSV] 全量  -> {path}")

    if output_dir and ts_str:
        for cyc in range(n_cycles):
            lo  = cyc * cycle_len
            hi  = (cyc + 1) * cycle_len
            cyc_path = os.path.join(output_dir,
                                    f"iv_cycle{cyc+1}_{ts_str}.csv")
            with open(cyc_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for k in range(lo, hi):
                    w.writerow(row(k))
            print(f"[CSV] Cycle {cyc+1} -> {cyc_path}")


# ──────────────────────────────────────────────────────
def _icc_reference_lines(ax):
    """在 I-V 图上画非对称限流参考线 (正负不同值)"""
    if SET_POLARITY == "positive":
        icc_pos, icc_neg = I_CC_SET, I_CC_RESET
    else:
        icc_pos, icc_neg = I_CC_RESET, I_CC_SET

    ax.axhline( icc_pos*1e6, color="#FF5722", lw=0.9, linestyle="--",
               alpha=0.7,
               label=f"+I_cc = {icc_pos*1e6:.0f} µA" if icc_pos < 1e-3
                     else f"+I_cc = {icc_pos*1e3:.1f} mA")
    ax.axhline(-icc_neg*1e6, color="#FF9800", lw=0.9, linestyle="--",
               alpha=0.7,
               label=f"-I_cc = {icc_neg*1e6:.0f} µA" if icc_neg < 1e-3
                     else f"-I_cc = {icc_neg*1e3:.1f} mA")


# ──────────────────────────────────────────────────────
def plot_individual_cycles(voltages, currents, resistance,
                           n_cycles, cycle_len,
                           v_max, v_min, output_dir, ts_str):
    v_all = voltages
    i_all = currents * 1e6
    r_all = resistance / 1e3

    for cyc in range(n_cycles):
        lo  = cyc * cycle_len
        hi  = (cyc + 1) * cycle_len

        v_c = v_all[lo:hi]
        i_c = i_all[lo:hi]
        r_c = r_all[lo:hi]
        x_c = np.arange(cycle_len)
        col = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0F1117")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1A1D27")
            ax.tick_params(colors="#AAAAAA", labelsize=9)
            for sp in ax.spines.values(): sp.set_color("#444")

        seg_labels = [
            (0,             cycle_len//4,   "#69F0AE", "0 → V_max"),
            (cycle_len//4,  cycle_len//2,   "#FFD54F", "V_max → 0"),
            (cycle_len//2,  3*cycle_len//4, "#80DEEA", "0 → V_min"),
            (3*cycle_len//4, cycle_len,     "#EF9A9A", "V_min → 0"),
        ]
        for lo2, hi2, seg_col, seg_label in seg_labels:
            ax1.plot(v_c[lo2:hi2], i_c[lo2:hi2],
                     color=seg_col, lw=2.0, alpha=0.9,
                     label=seg_label, zorder=3)

        for lo2, hi2, seg_col, _ in seg_labels:
            mid = (lo2 + hi2) // 2
            if mid + 1 >= cycle_len: continue
            dv = v_c[mid+1] - v_c[mid]
            di = i_c[mid+1] - i_c[mid]
            if abs(dv) + abs(di) < 1e-9: continue
            ax1.annotate("", xy=(v_c[mid+1], i_c[mid+1]),
                xytext=(v_c[mid], i_c[mid]),
                arrowprops=dict(arrowstyle="-|>", color=seg_col,
                                lw=1.0, mutation_scale=10))

        ax1.scatter([v_c[0]], [i_c[0]], color="#FFFFFF", s=70,
                    zorder=6, marker="o", edgecolors=col, linewidths=1.5,
                    label="Start")

        _icc_reference_lines(ax1)
        ax1.axhline(0, color="#555", lw=0.5)
        ax1.axvline(0, color="#555", lw=0.5)

        vp = max((v_c.max()-v_c.min())*0.08, 0.2)
        ip = max((i_c.max()-i_c.min())*0.08, 10.0)
        ax1.set_xlim(v_c.min()-vp, v_c.max()+vp)
        ax1.set_ylim(i_c.min()-ip, i_c.max()+ip)
        ax1.set_xlabel("Voltage (V)",  color="#CCCCCC", fontsize=10)
        ax1.set_ylabel("Current (µA)", color="#CCCCCC", fontsize=10)
        ax1.set_title(f"I–V  |  Cycle {cyc+1}",
                      color="#EEEEEE", fontsize=11, pad=8)
        ax1.legend(facecolor="#1A1D27", edgecolor="#444",
                   labelcolor="#CCCCCC", fontsize=7.5, loc="best")
        ax1.grid(True, color="#2A2D37", lw=0.5, zorder=0)

        fin = np.isfinite(r_c)
        ax2.scatter(x_c[fin], r_c[fin], c=col, s=6, alpha=0.75, zorder=3)

        win = max(5, int(cycle_len * 0.08)); half = win // 2
        sm = np.full(cycle_len, np.nan)
        for k in range(cycle_len):
            ch = r_c[max(0, k-half):min(cycle_len, k+half+1)]
            vc2 = ch[np.isfinite(ch)]
            if len(vc2): sm[k] = np.nanmedian(vc2)
        ax2.plot(x_c, sm, color=col, lw=2.0, alpha=0.9, label="Median smooth")

        for sep in [cycle_len//4, cycle_len//2, 3*cycle_len//4]:
            ax2.axvline(sep, color="#555", lw=0.7, linestyle=":", alpha=0.7)

        fin_r = r_c[fin]
        if len(fin_r) >= 5:
            rlo = np.percentile(fin_r, 5)
            rhi = np.percentile(fin_r, 95)
            ax2.axhline(rlo, color="#4DB6AC", lw=1.0, linestyle=":", alpha=0.9)
            ax2.axhline(rhi, color="#FFB74D", lw=1.0, linestyle=":", alpha=0.9)
            ax2.text(1, rlo*1.15, f"P5  ≈ {rlo:.1f} kΩ",
                     color="#4DB6AC", fontsize=7.5)
            ax2.text(1, rhi*1.15, f"P95 ≈ {rhi:.1f} kΩ",
                     color="#FFB74D", fontsize=7.5)
            if rlo > 0:
                ax2.text(0.97, 0.04, f"P95/P5 ≈ {rhi/rlo:.1f}×",
                         transform=ax2.transAxes, color="#CE93D8",
                         fontsize=9, ha="right",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="#1A1D27", ec="#9C27B0", alpha=0.85))

        ax2.set_yscale("log")
        ax2.set_xlabel("Sample Index (within cycle)",
                       color="#CCCCCC", fontsize=10)
        ax2.set_ylabel("Resistance (kΩ)", color="#CCCCCC", fontsize=10)
        ax2.set_title(f"Resistance  |  Cycle {cyc+1}",
                      color="#EEEEEE", fontsize=11, pad=8)
        ax2.legend(facecolor="#1A1D27", edgecolor="#444",
                   labelcolor="#CCCCCC", fontsize=8, loc="upper right")
        ax2.grid(True, color="#2A2D37", lw=0.5, which="both", zorder=0)

        icc_str = (f"SET={I_CC_SET*1e6:.0f}µA / "
                   f"RESET={I_CC_RESET*1e3:.1f}mA")
        fig.suptitle(
            f"Ni/AlOx/TiOx/Ti   Cycle {cyc+1}/{n_cycles}   "
            f"V: {v_min}~{v_max} V   {cycle_len} pts   {icc_str}",
            color="#DDDDDD", fontsize=9, y=1.01)

        plt.tight_layout()
        img_path = os.path.join(output_dir,
                                f"iv_cycle{cyc+1}_{ts_str}.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[PNG] Cycle {cyc+1} -> {img_path}")


# ──────────────────────────────────────────────────────
def plot_and_save(voltages, currents, resistance,
                  n_cycles, cycle_len,
                  v_max, v_min, img_path):

    v   = voltages
    i   = currents * 1e6
    n   = len(v)
    idx = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        for sp in ax.spines.values(): sp.set_color("#444")

    LINESTYLES = ["-", "--", "-.", (0,(3,1,1,1)), (0,(5,1))]
    LW         = [1.8, 1.8, 1.8, 1.6, 1.6]

    for cyc in range(n_cycles):
        lo  = cyc * cycle_len
        hi  = (cyc + 1) * cycle_len
        v_c = v[lo:hi]
        i_c = i[lo:hi]
        col = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]
        ls  = LINESTYLES[cyc % len(LINESTYLES)]
        lw  = LW[cyc % len(LW)]

        ax1.plot(v_c, i_c, color=col, lw=lw, linestyle=ls,
                 alpha=0.85, zorder=2+cyc*0.1)
        ax1.scatter([v_c[0]], [i_c[0]], color=col, s=55, zorder=6,
                    marker="o", edgecolors="white", linewidths=0.6)

        m = len(v_c)
        for ap in np.linspace(0, m-2, 4, dtype=int):
            dv = v_c[ap+1] - v_c[ap]
            di = i_c[ap+1] - i_c[ap]
            if abs(dv) + abs(di) < 1e-9: continue
            ax1.annotate("", xy=(v_c[ap+1], i_c[ap+1]),
                xytext=(v_c[ap], i_c[ap]),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=0.8, mutation_scale=9))

        ax1.plot([], [], color=col, lw=lw, linestyle=ls,
                 label=f"Cycle {cyc+1}")

    _icc_reference_lines(ax1)
    ax1.axhline(0, color="#555", lw=0.5)
    ax1.axvline(0, color="#555", lw=0.5)

    vp = max((v.max()-v.min())*0.08, 0.2)
    ip = max((i.max()-i.min())*0.08, 10.0)
    ax1.set_xlim(v.min()-vp, v.max()+vp)
    ax1.set_ylim(i.min()-ip, i.max()+ip)
    ax1.set_xlabel("Voltage (V)",   color="#CCCCCC", fontsize=10)
    ax1.set_ylabel("Current (µA)",  color="#CCCCCC", fontsize=10)
    ax1.set_title("I–V Characteristics", color="#EEEEEE", fontsize=11, pad=8)
    ax1.legend(facecolor="#1A1D27", edgecolor="#444",
               labelcolor="#CCCCCC", fontsize=8.5, loc="best")
    ax1.grid(True, color="#2A2D37", lw=0.5, zorder=0)

    r_k = resistance / 1e3
    for cyc in range(n_cycles):
        lo   = cyc * cycle_len
        hi   = (cyc + 1) * cycle_len
        r_c  = r_k[lo:hi]
        x_c  = idx[lo:hi]
        col  = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]
        fin  = np.isfinite(r_c)
        ax2.scatter(x_c[fin], r_c[fin], c=col, s=5, alpha=0.7, zorder=3)

        win = max(5, int(cycle_len * 0.08))
        half = win // 2
        sm = np.full(cycle_len, np.nan)
        for k in range(cycle_len):
            ch = r_c[max(0, k-half):min(cycle_len, k+half+1)]
            vc = ch[np.isfinite(ch)]
            if len(vc): sm[k] = np.nanmedian(vc)
        ax2.plot(x_c, sm, color=col, lw=1.5, alpha=0.9, label=f"Cycle {cyc+1}")

    for cyc in range(1, n_cycles):
        ax2.axvline(cyc*cycle_len, color="#444", lw=0.8,
                    linestyle="--", alpha=0.6)

    fin_all = r_k[np.isfinite(r_k)]
    if len(fin_all) >= 5:
        rlo = np.percentile(fin_all, 5)
        rhi = np.percentile(fin_all, 95)
        ax2.axhline(rlo, color="#4DB6AC", lw=1.0, linestyle=":", alpha=0.9)
        ax2.axhline(rhi, color="#FFB74D", lw=1.0, linestyle=":", alpha=0.9)
        ax2.text(2, rlo*1.15, f"P5  ≈ {rlo:.1f} kΩ",
                 color="#4DB6AC", fontsize=7.5)
        ax2.text(2, rhi*1.15, f"P95 ≈ {rhi:.1f} kΩ",
                 color="#FFB74D", fontsize=7.5)
        if rlo > 0:
            ax2.text(0.97, 0.04, f"P95/P5 ≈ {rhi/rlo:.1f}×",
                     transform=ax2.transAxes, color="#CE93D8",
                     fontsize=9.5, ha="right",
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="#1A1D27", ec="#9C27B0", alpha=0.85))

    ax2.set_yscale("log")
    ax2.set_xlabel("Sample Index",    color="#CCCCCC", fontsize=10)
    ax2.set_ylabel("Resistance (kΩ)", color="#CCCCCC", fontsize=10)
    ax2.set_title("Resistance vs. Sample Index",
                  color="#EEEEEE", fontsize=11, pad=8)
    ax2.legend(facecolor="#1A1D27", edgecolor="#444",
               labelcolor="#CCCCCC", fontsize=8.5)
    ax2.grid(True, color="#2A2D37", lw=0.5, which="both", zorder=0)

    icc_str = f"SET={I_CC_SET*1e6:.0f}µA / RESET={I_CC_RESET*1e3:.1f}mA"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"Ni/AlOx/TiOx/Ti   V: {v_min}~{v_max} V   "
        f"{n_cycles} cycles × {cycle_len} pts   "
        f"{icc_str}   NPLC={NPLC}   {ts}",
        color="#DDDDDD", fontsize=9, y=1.01)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] 已保存: {img_path}")
    plt.show()


# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 文件名反映非对称限流
    run_name = (
        f"V{V_MIN}_{V_MAX}_"
        f"cyc{N_CYCLES}_"
        f"pts{N_POINTS}_"
        f"SET{int(I_CC_SET*1e6)}uA_"
        f"RST{I_CC_RESET*1e3:.0f}mA_"
        f"NPLC{NPLC}_"
        f"{ts_str}"
    )
    RUN_DIR  = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"[输出] {RUN_DIR}")

    csv_path = os.path.join(RUN_DIR, f"iv_sweep_{ts_str}.csv")
    img_path = os.path.join(RUN_DIR, f"iv_sweep_{ts_str}.png")

    cycle_len  = N_POINTS * 4
    total      = cycle_len * N_CYCLES
    step_delay = max(0.02, NPLC / 60 * 1.5 + 0.02)

    print("=" * 56)
    print("  IV Sweep 参数 (非对称限流版 v4)")
    print("=" * 56)
    print(f"  V_MAX        = {V_MAX} V")
    print(f"  V_MIN        = {V_MIN} V")
    print(f"  N_POINTS     = {N_POINTS}  (每 1/4 段)")
    print(f"  N_CYCLES     = {N_CYCLES}")
    print(f"  SET 极性     = {SET_POLARITY}")
    print(f"  I_CC_SET     = {I_CC_SET*1e6:.0f} µA   (限流, 控制细丝)")
    print(f"  I_CC_RESET   = {I_CC_RESET*1e3:.1f} mA   (放开, 拉断细丝)")
    if I_CC_RESET >= 1.0:
        print("  [WARNING] I_CC_RESET >= 1 A. 5000000e-6 means 5 A, not 5 mA.")
        print("            For 5 mA use 5000e-6 or 5e-3.")
    print(f"  NPLC         = {NPLC}")
    print(f"  总点数       = {total}")
    print(f"  预计耗时     ≈ {total*step_delay:.0f} s")
    print(f"  输出目录     = {RUN_DIR}")
    print("=" * 56)
    if SET_POLARITY == "positive":
        print("  限流方向: 正压段(SET)限流 / 负压段(RESET)放开")
    else:
        print("  限流方向: 负压段(SET)限流 / 正压段(RESET)放开")
    print("  (!) RESET电流较大,首次从小往大试(1→10→50mA),防过生长损伤")
    print("=" * 56)

    ans = input("\n确认开始扫描? (Enter / q 退出): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)
    init_smu(inst, I_CC_SET, NPLC)   # 初始用 SET 限流(保守)

    voltages, currents, timestamps, cycle_len, icc_used = run_sweep(
        inst, V_MAX, V_MIN, N_POINTS, N_CYCLES, NPLC)
    inst.close()

    resistance = compute_resistance_with_icc_hold(voltages, currents, icc_used)

    fin = resistance[np.isfinite(resistance)]
    print(f"\n  点数     : {len(voltages)}")
    print(f"  电压范围 : {voltages.min():.4f} ~ {voltages.max():.4f} V")
    print(f"  电流峰值 : {np.abs(currents).max()*1e6:.3f} µA")
    if len(fin):
        print(f"  电阻范围 : {fin.min()/1e3:.3f} ~ {fin.max()/1e3:.3f} kΩ")

    # ── RESET 诊断: 检查 RESET 方向是否出现电流突降 ──
    print("\n" + "─" * 56)
    print("  RESET 诊断")
    print("─" * 56)
    one_len = cycle_len
    v_last = voltages[-one_len:]
    i_last = currents[-one_len:]
    if SET_POLARITY == "positive":
        # RESET 在负压段
        reset_mask = v_last < -0.1
    else:
        reset_mask = v_last > 0.1

    if reset_mask.sum() > 5:
        v_r = v_last[reset_mask]
        i_r = np.abs(i_last[reset_mask])
        # 关键: 比较"同等电压幅度"下电流是否下降, 排除回零导致的自然下降
        # 真RESET = 电流在某固定高电压下突然变小(电阻突增)
        # 用电阻判据更可靠: RESET段电阻是否出现明显跳增
        with np.errstate(divide="ignore", invalid="ignore"):
            r_r = np.where(i_r > 1e-11, np.abs(v_r) / i_r, np.nan)
        r_r_fin = r_r[np.isfinite(r_r)]
        if len(r_r_fin) >= 5:
            r_lo = np.percentile(r_r_fin, 10)   # 低阻(RESET前)
            r_hi = np.percentile(r_r_fin, 90)   # 高阻(RESET后)
            ratio = r_hi / r_lo if r_lo > 0 else 1.0
            print(f"  RESET段电阻: 低{r_lo/1e3:.2f}kΩ → 高{r_hi/1e3:.2f}kΩ  "
                  f"(比值 {ratio:.2f}×)")
            if ratio >= 3.0:
                print("  [OK] RESET段电阻明显跳增(>3×) → 可能存在真RESET!")
                print("       → 看图确认是否突变; 若是: 之前限流压制了RESET")
            elif ratio >= 1.5:
                print("  (~) RESET段电阻有小幅上升(1.5-3×) → 部分RESET/不完全")
                print("      → 可尝试更大RESET电流或更负电压")
            else:
                print("  (!) RESET段电阻几乎不变(<1.5×) → RESET失败")
                print("      → 放开限流仍断不开, 坐实Ti电极吸氧问题")
                print("      → 需换Pt底电极 或 Ti上加AlOx阻挡层(见笔记)")
        else:
            print("  [警告] RESET段有效电阻点不足, 无法判定")
    else:
        print("  [警告] RESET段采样点不足")
    print("─" * 56)

    save_csv(voltages, currents, timestamps, resistance, icc_used,
             cycle_len, csv_path, RUN_DIR, ts_str)

    print("\n[绘图] 生成各周期独立图像...")
    plot_individual_cycles(voltages, currents, resistance,
                           N_CYCLES, cycle_len,
                           V_MAX, V_MIN, RUN_DIR, ts_str)

    print("[绘图] 生成汇总图像...")
    plot_and_save(voltages, currents, resistance,
                  N_CYCLES, cycle_len, V_MAX, V_MIN, img_path)


if __name__ == "__main__":
    main()
