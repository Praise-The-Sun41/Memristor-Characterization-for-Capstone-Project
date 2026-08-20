"""
pulse_ltp_ltd.py — 忆阻器脉冲调制 (LTP/LTD) 实验  v1
=====================================================
器件: Ni / AlOx / TiOx / Ti  (极性可配置)

实验目的:
  验证器件能否作为模拟突触 —— 用电压脉冲渐进、可控地调制电导
    LTP (增强): 一连串同极性脉冲 → 电导 G 单调上升
    LTD (抑制): 反极性脉冲       → 电导 G 单调下降

测量原子操作 (一次"写-读"):
  施加写脉冲 → 回零 → 弛豫等待 → 小信号读电导 → 回零

★ 重要约束 ★
  Keithley 2450 是 SMU,非脉冲发生器,最小可靠脉宽约 1-5 ms。
  本脚本的"脉冲"是 ms 级软件定时准脉冲,作为 ns 脉冲行为的慢速代理。
  数据命名含 "msPulse" 以示区分。

依赖: pip install pyvisa pyvisa-py numpy matplotlib
"""

import os
import sys
import time
import argparse
from datetime import datetime

import platform
import matplotlib
_fonts = {"Windows": "Microsoft YaHei", "Darwin": "PingFang SC",
          "Linux": "WenQuanYi Micro Hei"}
matplotlib.rcParams["font.family"]        = _fonts.get(platform.system(), "DejaVu Sans")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use("TkAgg")   # 无显示器改 "Agg"

import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════
#  ★ 可编辑参数区 ★
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = "USB0::0x05E6::0x2450::04437634::INSTR"

# ── 脉冲幅度 (带符号,极性可配置) ────────────────────
#   默认: 负压增强 (LTP), 正压抑制 (LTD) —— 同 An et al.
#   若器件极性相反,把符号对调即可
V_LTP      = 0.5     # 增强脉冲幅度 (V)
V_LTD      = -0.5     # 抑制脉冲幅度 (V)

# ── 脉冲时序 ──────────────────────────────────────
T_PULSE    = 100e-3     # 脉宽 (s)，2450 可靠下限约 1-5 ms
T_RELAX    = 100e-3    # 写脉冲后到读取的弛豫等待 (s)

# ── 脉冲数量 ──────────────────────────────────────
N_LTP      = 5       # 增强脉冲数
N_LTD      = 5       # 抑制脉冲数
N_CYCLES   = 1        # LTP-LTD 循环重复次数 (v1 先用 1)

# ── 初始 RESET (诊断用) ───────────────────────────
#   目的: LTP 段前先把器件拉到高阻态,给 LTP 留出上升空间
#   注意: 用比 LTD 脉冲更强的单次电压 (脉冲幅度往往不足以完全 reset)
#   诊断逻辑:
#     RESET 后 LTP 能单调爬升  → 之前是"起点卡死",器件有救
#     RESET 后仍 2% 随机抖动   → 动态范围不足,需先改器件
ENABLE_INIT_RESET = True   # 是否启用初始 RESET
V_INIT_RESET   = -1.5      # 初始 RESET 电压 (V)，用 IV 扫描里的 RESET 幅度
T_INIT_RESET   = 50e-3     # 初始 RESET 脉宽 (s)，较长确保 reset 充分
N_INIT_RESET   = 20         # 初始 RESET 脉冲数
RESET_MIN_DROP = 0.1       # 判据: RESET 后电导下降不足此比例则提示"reset不动器件"

# ── 读取参数 ──────────────────────────────────────
V_READ     = 0.1      # 读电压 (V)，远小于开关电压，非破坏性
N_READ_AVG = 3       # 每次读取的采样平均次数
T_READ_SETTLE = 5e-3  # 读取稳定等待 (s)

# ── 保护 ──────────────────────────────────────────
I_CC       = 50e-6   # 顺应电流 (A)
NPLC       = 0.1      # 积分时间 (短，加快脉冲读取)
# ══════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Pulse_LTP_LTD")


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
        for r in rm.list_resources():
            print(f"  可用: {r}")
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
        print(f"[初始化] I_cc = {i_cc*1e6:.0f} µA, NPLC = {nplc}, 完成")


# ──────────────────────────────────────────────────────
#  单次读电流 (返回电流 A)
# ──────────────────────────────────────────────────────
def read_current_once(inst) -> float:
    try:
        raw   = inst.query(":READ?").strip()
        parts = raw.split(",")
        return float(parts[1]) if len(parts) >= 2 else float(parts[0])
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────
#  读电导 (小信号, N 次平均) —— 非破坏性
#  返回 (G 西门子, R 欧姆, I 安培)
# ──────────────────────────────────────────────────────
def read_conductance(inst, v_read: float, n_avg: int,
                     settle: float) -> tuple:
    inst.write(f":SOUR:VOLT:LEV {v_read:.4f}")
    inst.write(":OUTP ON")
    time.sleep(settle)

    currents = []
    for _ in range(n_avg):
        i = read_current_once(inst)
        if not np.isnan(i):
            currents.append(i)
        time.sleep(0.001)

    # 读完回零 (非破坏: 不关输出, 仅置零电压)
    inst.write(":SOUR:VOLT:LEV 0")

    if not currents:
        return float("nan"), float("inf"), float("nan")

    i_mean = float(np.mean(currents))
    if abs(i_mean) > 1e-13:
        R = abs(v_read / i_mean)
        G = 1.0 / R
    else:
        R = float("inf")
        G = 0.0
    return G, R, i_mean


# ──────────────────────────────────────────────────────
#  施加单个写脉冲 → 回零 → 弛豫
# ──────────────────────────────────────────────────────
def apply_pulse(inst, v_pulse: float, t_pulse: float, t_relax: float):
    inst.write(f":SOUR:VOLT:LEV {v_pulse:.4f}")
    inst.write(":OUTP ON")
    time.sleep(t_pulse)
    inst.write(":SOUR:VOLT:LEV 0")   # 回零
    time.sleep(t_relax)              # 弛豫等待


# ──────────────────────────────────────────────────────
#  初始 RESET (诊断): 把器件拉到高阻态, 量化拉低了多少电导
#  返回 (G_before, G_after, 诊断字符串)
# ──────────────────────────────────────────────────────
def initial_reset(inst) -> tuple:
    print(f"\n{'─'*56}")
    print(f"  初始 RESET (诊断)")
    print(f"  V = {V_INIT_RESET:+.2f} V × {N_INIT_RESET} 脉冲, "
          f"脉宽 {T_INIT_RESET*1e3:.0f} ms")
    print(f"{'─'*56}")

    G_before, R_before, _ = read_conductance(inst, V_READ,
                                             N_READ_AVG, T_READ_SETTLE)
    print(f"  RESET 前: G = {G_before*1e6:.3f} µS   R = {R_before/1e3:.2f} kΩ")

    for k in range(N_INIT_RESET):
        apply_pulse(inst, V_INIT_RESET, T_INIT_RESET, T_RELAX)
        G, R, _ = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
        print(f"  reset #{k}: G = {G*1e6:.3f} µS   R = {R/1e3:.2f} kΩ")

    G_after, R_after, _ = read_conductance(inst, V_READ,
                                           N_READ_AVG, T_READ_SETTLE)
    print(f"  RESET 后: G = {G_after*1e6:.3f} µS   R = {R_after/1e3:.2f} kΩ")

    # ── 诊断判据 ──────────────────────────────────────
    if G_before > 0:
        drop = (G_before - G_after) / G_before
    else:
        drop = 0.0

    if drop >= RESET_MIN_DROP:
        diag = (f"[OK] 电导下降 {drop*100:.1f}%, RESET 有效, "
                f"LTP 有上升空间")
    else:
        diag = (f"(!) 电导仅变化 {drop*100:.1f}% (<{RESET_MIN_DROP*100:.0f}%), "
                f"此电压可能 reset 不动器件 —— 检查 V_INIT_RESET / 器件窗口")
    print(f"  诊断: {diag}")

    return G_before, G_after, diag


# ──────────────────────────────────────────────────────
#  一个完整 LTP-LTD 循环
#  返回 dict: phase/pulse_idx/G/R/I 列表
# ──────────────────────────────────────────────────────
def run_ltp_ltd_cycle(inst, cycle_idx: int) -> dict:
    records = []   # 每条: (cycle, phase, pulse_idx, v_pulse, G, R, I)

    print(f"\n{'─'*56}")
    print(f"  Cycle {cycle_idx+1} / {N_CYCLES}")
    print(f"{'─'*56}")

    # ── LTP 段 ────────────────────────────────────────
    print(f"  [LTP] V_pulse = {V_LTP:+.2f} V × {N_LTP} 脉冲")
    print(f"  {'#':>4}  {'G (µS)':>10}  {'R (kΩ)':>10}")
    for k in range(N_LTP):
        apply_pulse(inst, V_LTP, T_PULSE, T_RELAX)
        G, R, I = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
        records.append((cycle_idx, "LTP", k, V_LTP, G, R, I))
        if k % 4 == 0 or k == N_LTP - 1:
            print(f"  {k:>4}  {G*1e6:>10.3f}  {R/1e3:>10.2f}")

    # ── LTD 段 ────────────────────────────────────────
    print(f"  [LTD] V_pulse = {V_LTD:+.2f} V × {N_LTD} 脉冲")
    print(f"  {'#':>4}  {'G (µS)':>10}  {'R (kΩ)':>10}")
    for k in range(N_LTD):
        apply_pulse(inst, V_LTD, T_PULSE, T_RELAX)
        G, R, I = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
        records.append((cycle_idx, "LTD", k, V_LTD, G, R, I))
        if k % 4 == 0 or k == N_LTD - 1:
            print(f"  {k:>4}  {G*1e6:>10.3f}  {R/1e3:>10.2f}")

    return records





# ──────────────────────────────────────────────────────
#  保存 CSV
# ──────────────────────────────────────────────────────
def save_csv(all_records: list, path: str):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "phase", "pulse_idx", "v_pulse_V",
                    "G_S", "G_uS", "R_Ohm", "I_A"])
        for (cyc, phase, idx, vp, G, R, I) in all_records:
            w.writerow([
                cyc, phase, idx, f"{vp:.4f}",
                f"{G:.6e}", f"{G*1e6:.4f}",
                f"{R:.2f}" if np.isfinite(R) else "inf",
                f"{I:.6e}",
            ])
    print(f"\n[CSV] 已保存: {path}")


# ──────────────────────────────────────────────────────
#  绘图: G vs 脉冲序号 (LTP 升 + LTD 降)
# ──────────────────────────────────────────────────────
def plot_ltp_ltd(all_records: list, img_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="#AAAAAA", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#444")

    cycles = sorted(set(r[0] for r in all_records))
    cmap_ltp = ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9"]
    cmap_ltd = ["#FF5252", "#FF7043", "#FF8A65", "#FFAB91"]

    global_x = 0
    for cyc in cycles:
        ltp = [r for r in all_records if r[0] == cyc and r[1] == "LTP"]
        ltd = [r for r in all_records if r[0] == cyc and r[1] == "LTD"]

        x_ltp = list(range(global_x, global_x + len(ltp)))
        g_ltp = [r[4]*1e6 for r in ltp]
        global_x += len(ltp)

        x_ltd = list(range(global_x, global_x + len(ltd)))
        g_ltd = [r[4]*1e6 for r in ltd]
        global_x += len(ltd)

        c_ltp = cmap_ltp[cyc % len(cmap_ltp)]
        c_ltd = cmap_ltd[cyc % len(cmap_ltd)]

        ax.plot(x_ltp, g_ltp, "-o", color=c_ltp, ms=4, lw=1.5,
                label=f"LTP cyc{cyc+1}" if len(cycles) > 1 else "LTP (增强)")
        ax.plot(x_ltd, g_ltd, "-s", color=c_ltd, ms=4, lw=1.5,
                label=f"LTD cyc{cyc+1}" if len(cycles) > 1 else "LTD (抑制)")

    ax.set_xlabel("Pulse number (#)", color="#CCCCCC", fontsize=11)
    ax.set_ylabel("Conductance G (µS)", color="#CCCCCC", fontsize=11)
    ax.set_title(
        f"LTP / LTD 脉冲调制   "
        f"V_LTP={V_LTP:+.1f}V  V_LTD={V_LTD:+.1f}V  "
        f"t_pulse={T_PULSE*1e3:.0f}ms  V_read={V_READ}V",
        color="#EEEEEE", fontsize=10, pad=10)
    ax.legend(facecolor="#1A1D27", edgecolor="#444",
              labelcolor="#CCCCCC", fontsize=9)
    ax.grid(True, color="#2A2D37", lw=0.5)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f"Ni/AlOx/TiOx/Ti   {ts}   (ms-pulse proxy)",
             color="#555", fontsize=7, ha="right")

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] 已保存: {img_path}")
    plt.show()


# ──────────────────────────────────────────────────────
#  主程序
# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="忆阻器脉冲调制 LTP/LTD v1")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    parser.add_argument("--dev", default="DEV",
                        help="器件标签,用于文件夹命名")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.dev}_"
        f"LTP{V_LTP:+.1f}_LTD{V_LTD:+.1f}_"
        f"msPulse{int(T_PULSE*1000)}_"
        f"N{N_LTP}x{N_LTD}_"
        f"{ts_str}"
    )
    RUN_DIR = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(RUN_DIR, exist_ok=True)
    csv_path = os.path.join(RUN_DIR, f"ltp_ltd_{ts_str}.csv")
    img_path = os.path.join(RUN_DIR, f"ltp_ltd_{ts_str}.png")

    # 参数确认
    print("=" * 56)
    print("  脉冲调制 LTP/LTD 实验  v1  (流程验证版)")
    print("=" * 56)
    print(f"  V_LTP (增强)  = {V_LTP:+.2f} V")
    print(f"  V_LTD (抑制)  = {V_LTD:+.2f} V")
    print(f"  脉宽          = {T_PULSE*1e3:.1f} ms  (ms级准脉冲)")
    print(f"  弛豫等待      = {T_RELAX*1e3:.1f} ms")
    print(f"  脉冲数        = LTP {N_LTP} + LTD {N_LTD}")
    print(f"  循环数        = {N_CYCLES}")
    print(f"  读电压        = {V_READ} V  ({N_READ_AVG}次平均)")
    print(f"  I_cc          = {I_CC*1e6:.0f} µA")
    n_pulses = (N_LTP + N_LTD) * N_CYCLES
    est_s    = n_pulses * (T_PULSE + T_RELAX + T_READ_SETTLE + 0.02)
    print(f"  脉冲总数      = {n_pulses}   耗时估计 ≈ {est_s:.0f} s")
    print(f"  器件标签      = {args.dev}")
    print(f"  输出目录      = {RUN_DIR}")
    print("=" * 56)
    print("  注: 2450为SMU,ms级准脉冲,作为ns脉冲行为的慢速代理")
    print("=" * 56)

    ans = input("\n确认开始? (Enter / q 退出): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)
    init_smu(inst, I_CC, NPLC)

    # 初始电导基线
    G0, R0, I0 = read_conductance(inst, V_READ, N_READ_AVG, T_READ_SETTLE)
    print(f"\n[基线] 初始 G = {G0*1e6:.3f} µS   R = {R0/1e3:.2f} kΩ")

    # 初始 RESET (诊断)
    reset_diag = None
    if ENABLE_INIT_RESET:
        _, _, reset_diag = initial_reset(inst)

    all_records = []
    try:
        for cyc in range(N_CYCLES):
            recs = run_ltp_ltd_cycle(inst, cyc)
            all_records.extend(recs)
    except KeyboardInterrupt:
        print("\n[中断] 用户终止,保存已采集数据...")
    finally:
        inst.write(":SOUR:VOLT:LEV 0")
        inst.write(":OUTP OFF")
        inst.close()

    if not all_records:
        print("[结果] 无数据"); return

    # 汇总
    g_all = [r[4]*1e6 for r in all_records]
    print("\n" + "=" * 56)
    print("  汇总")
    print("=" * 56)
    print(f"  初始 G        : {G0*1e6:.3f} µS")
    print(f"  G 范围        : {min(g_all):.3f} ~ {max(g_all):.3f} µS")
    if min(g_all) > 0:
        ratio = max(g_all) / min(g_all)
        print(f"  G_max/G_min   : {ratio:.2f}×")
    print(f"  采集点数      : {len(all_records)}")

    # ── LTP/LTD 单调性诊断 ────────────────────────────
    ltp_g = [r[4]*1e6 for r in all_records if r[1] == "LTP"]
    ltd_g = [r[4]*1e6 for r in all_records if r[1] == "LTD"]
    if len(ltp_g) > 2 and len(ltd_g) > 2:
        # 净变化 vs 抖动幅度
        ltp_net = ltp_g[-1] - ltp_g[0]          # LTP 净上升
        ltd_net = ltd_g[0] - ltd_g[-1]          # LTD 净下降
        noise   = float(np.std(np.diff(ltp_g + ltd_g)))  # 点间抖动
        span    = max(g_all) - min(g_all)
        print(f"  LTP 净变化    : {ltp_net:+.3f} µS")
        print(f"  LTD 净变化    : {ltd_net:+.3f} µS (正=下降)")
        print(f"  点间抖动 std  : {noise:.3f} µS")
        print("  " + "-" * 50)
        # 诊断结论
        if ltp_net > 2 * noise and ltd_net > 2 * noise:
            print("  [OK] LTP上升+LTD下降均显著 > 抖动 → 器件有调制能力")
            print("       之前是起点问题, 加RESET后有效")
        else:
            print("  (!) LTP/LTD 净变化未明显超过抖动")
            print("      → 动态范围不足, 坐实需先改器件(开关比仍是瓶颈)")
            print("      → 见笔记: Pt电极 / AlOx位置 / 厚度梯度")
    if reset_diag:
        print("  " + "-" * 50)
        print(f"  RESET 诊断    : {reset_diag}")
    print("=" * 56)

    save_csv(all_records, csv_path)
    plot_ltp_ltd(all_records, img_path)


if __name__ == "__main__":
    main()