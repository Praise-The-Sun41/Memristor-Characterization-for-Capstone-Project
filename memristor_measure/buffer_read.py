"""
read_buffer.py — Keithley 2450 Buffer 数据读出脚本
===================================================
功能:
  1. 连接 Keithley 2450
  2. 读出指定 buffer 中的所有数据（电压、电流、时间戳）
  3. 计算电阻
  4. 绘制 I-V 曲线 + 电阻-索引图，保存为 PNG
  5. 保存原始数据为 CSV

依赖:
    pip install pyvisa pyvisa-py numpy matplotlib

用法:
    python read_buffer.py
    python read_buffer.py --address "GPIB0::24::INSTR"
    python read_buffer.py --buffer defbuffer2 --output my_data
"""

import argparse
import time
import sys
import os
from datetime import datetime
import numpy as np
import matplotlib
import platform
_fonts = {'Windows': 'Microsoft YaHei', 'Darwin': 'PingFang SC', 'Linux': 'WenQuanYi Micro Hei'}
matplotlib.rcParams['font.family'] = _fonts.get(platform.system(), 'DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use("TkAgg")          # 有显示器时用 TkAgg；无头环境改为 "Agg"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ══════════════════════════════════════════════════════
#  可调参数区
# ══════════════════════════════════════════════════════
DEFAULT_VISA_ADDRESS = "USB0::0x05E6::0x2450::04437634::INSTR"
DEFAULT_BUFFER       = "defbuffer1"   # 可选: defbuffer1 / defbuffer2 / 自定义名
OUTPUT_PREFIX        = "buffer_read/buffer_data"  # 输出文件前缀（不含扩展名）
# ══════════════════════════════════════════════════════


def connect(address: str):
    try:
        import pyvisa
    except ImportError:
        print("[ERROR] 缺少 pyvisa: pip install pyvisa pyvisa-py")
        sys.exit(1)

    rm = pyvisa.ResourceManager()
    print(f"[连接] {address}")
    try:
        inst = rm.open_resource(address)
    except Exception as e:
        print(f"[ERROR] 连接失败: {e}")
        print("可用资源:")
        for r in rm.list_resources():
            print(f"  {r}")
        sys.exit(1)

    inst.timeout           = 15000
    inst.write_termination = "\n"
    inst.read_termination  = "\n"

    idn = inst.query("*IDN?").strip()
    print(f"[连接] 成功 -> {idn}\n")
    return inst


def read_buffer(inst, buf_name: str) -> dict:
    """
    从指定 buffer 读取全部数据。
    返回 dict: {voltages, currents, timestamps, n_points}

    2450 SCPI 语法要点:
      - buffer 名必须用双引号: "defbuffer1"
      - 列名关键字: SOURce / READing / RELative
      - 单次查询所有列，减少往返次数，避免超时
    """
    # 查询 buffer 中实际点数（buffer名用双引号）
    n_str = inst.query(f':TRAC:ACT? "{buf_name}"').strip()
    try:
        n = int(float(n_str))
    except ValueError:
        print(f"[ERROR] 无法解析点数: '{n_str}'")
        sys.exit(1)

    if n == 0:
        print(f"[警告] Buffer '{buf_name}' 为空（0 个数据点）")
        sys.exit(0)

    print(f"[Buffer] '{buf_name}' 中共 {n} 个数据点，正在读取...")

    # 动态调整超时：每个点约 50 ms，最少 10 s
    inst.timeout = max(10000, n * 50 + 5000)

    # 单次读取三列（SOURce, READing, RELative），逗号分隔返回
    # 返回格式: s1, r1, t1, s2, r2, t2, ...
    raw = inst.query(
        f':TRAC:DATA? 1, {n}, "{buf_name}", SOURce, READing, RELative'
    ).strip()

    vals = [float(x) for x in raw.split(",")]

    if len(vals) != n * 3:
        print(f"[警告] 返回值数量 {len(vals)} 与预期 {n*3} 不符，尝试截断处理")
        n = len(vals) // 3
        vals = vals[:n * 3]

    arr       = np.array(vals).reshape(n, 3)
    voltages  = arr[:, 0]
    currents  = arr[:, 1]
    timestamps = arr[:, 2]

    print(f"[Buffer] 读取完成: {n} 点")

    # 查询仪器错误
    err = inst.query(":SYST:ERR?").strip()
    if not err.startswith("0"):
        print(f"[仪器警告] {err}")

    return {
        "voltages":   voltages,
        "currents":   currents,
        "timestamps": timestamps,
        "n_points":   n,
        "buf_name":   buf_name,
    }


def compute_resistance(voltages: np.ndarray,
                       currents: np.ndarray) -> np.ndarray:
    safe = np.abs(currents) > 1e-12
    resistance = np.where(safe,
                          np.abs(voltages / currents),
                          np.nan)
    return resistance


def save_csv(data: dict, resistance: np.ndarray, path: str):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "time_s", "voltage_V",
                    "current_A", "current_uA", "resistance_Ohm"])
        for k in range(data["n_points"]):
            r = resistance[k] if np.isfinite(resistance[k]) else ""
            w.writerow([
                k,
                f"{data['timestamps'][k]:.6f}",
                f"{data['voltages'][k]:.6f}",
                f"{data['currents'][k]:.6e}",
                f"{data['currents'][k]*1e6:.4f}",
                f"{r:.4f}" if r != "" else "inf",
            ])
    print(f"[CSV] 已保存: {path}")


def plot_and_save(data: dict, resistance: np.ndarray, img_path: str):
    v   = data["voltages"]
    i   = data["currents"] * 1e6    # µA
    t   = data["timestamps"]
    n   = data["n_points"]
    idx = np.arange(n)

    # ── 颜色渐变（按时间/索引） ───────────────────────────
    norm   = plt.Normalize(0, n - 1)
    cmap   = plt.get_cmap("plasma")
    colors = cmap(norm(idx))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")
    for ax in axes:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#444")

    ax1, ax2 = axes

    # ══ 左图: I-V 曲线（渐变色，箭头走势）══════════════
    pts  = np.array([v, i]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="plasma", norm=norm,
                          linewidth=1.5, zorder=2, alpha=0.9)
    lc.set_array(np.arange(len(segs)))
    ax1.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax1, pad=0.02)
    cbar.set_label("Sample Index", color="#AAAAAA", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#AAAAAA", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#AAAAAA")

    # 箭头（均匀分布 8 个）
    arrow_positions = np.linspace(0, n - 2, 8, dtype=int)
    for ap in arrow_positions:
        if ap + 1 >= n:
            continue
        dv = v[ap+1] - v[ap]
        di = i[ap+1] - i[ap]
        if abs(dv) + abs(di) < 1e-9:
            continue
        c = cmap(norm(ap))
        ax1.annotate("",
            xy=(v[ap+1], i[ap+1]),
            xytext=(v[ap], i[ap]),
            arrowprops=dict(arrowstyle="-|>", color=c,
                            lw=0.8, mutation_scale=9))

    # 起点 / 终点标记
    ax1.scatter([v[0]],  [i[0]],  color="#69F0AE", s=60, zorder=5,
                label=f"Start  ({v[0]:.3f} V, {i[0]:.2f} µA)", edgecolors="#fff", lw=0.5)
    ax1.scatter([v[-1]], [i[-1]], color="#FF4081", s=60, zorder=5,
                label=f"End    ({v[-1]:.3f} V, {i[-1]:.2f} µA)", edgecolors="#fff", lw=0.5)

    ax1.axhline(0, color="#555", lw=0.5)
    ax1.axvline(0, color="#555", lw=0.5)

    # 自动感知轴范围
    v_pad = max((v.max() - v.min()) * 0.1, 0.1)
    i_pad = max((i.max() - i.min()) * 0.1, 5.0)
    ax1.set_xlim(v.min() - v_pad, v.max() + v_pad)
    ax1.set_ylim(i.min() - i_pad, i.max() + i_pad)

    ax1.set_xlabel("Voltage (V)",    color="#CCCCCC", fontsize=10)
    ax1.set_ylabel("Current (µA)",   color="#CCCCCC", fontsize=10)
    ax1.set_title("I–V Curve  (color = time progression)",
                  color="#EEEEEE", fontsize=11, pad=8)
    ax1.legend(facecolor="#1A1D27", edgecolor="#555",
               labelcolor="#CCCCCC", fontsize=8, loc="best")
    ax1.grid(True, color="#2A2D37", lw=0.5, zorder=0)

    # ══ 右图: 电阻 vs 索引（对数坐标）══════════════════
    r_kohm = resistance / 1e3
    finite = np.isfinite(r_kohm)

    sc = ax2.scatter(idx[finite], r_kohm[finite],
                     c=idx[finite], cmap="plasma", norm=norm,
                     s=4, alpha=0.75, zorder=3)

    # 滑动中值平滑线（窗口 = 5% 点数，最少 5 点）
    win = max(5, int(n * 0.05))
    if np.sum(finite) >= win:
        from numpy.lib.stride_tricks import sliding_window_view
        r_valid = r_kohm.copy()
        r_valid[~finite] = np.nan
        # nanmedian 滚动
        half = win // 2
        smoothed = np.full(n, np.nan)
        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k + half + 1)
            chunk = r_kohm[lo:hi]
            valid_chunk = chunk[np.isfinite(chunk)]
            if len(valid_chunk) > 0:
                smoothed[k] = np.nanmedian(valid_chunk)
        ax2.plot(idx, smoothed, color="#FFD54F", lw=1.5,
                 alpha=0.85, label=f"滑动中值 (窗口={win})", zorder=4)

    # Ron / Roff 参考线（取有限值的 5th / 95th 百分位）
    if finite.sum() >= 5:
        r_fin = r_kohm[finite]
        r_low = np.percentile(r_fin, 5)
        r_hig = np.percentile(r_fin, 95)
        ax2.axhline(r_low, color="#4DB6AC", lw=1.0,
                    linestyle=":", alpha=0.9)
        ax2.text(n * 0.01, r_low * 1.15,
                 f"P5 ≈ {r_low:.2f} kΩ  (LRS参考)",
                 color="#4DB6AC", fontsize=7.5)
        ax2.axhline(r_hig, color="#FFB74D", lw=1.0,
                    linestyle=":", alpha=0.9)
        ax2.text(n * 0.01, r_hig * 1.15,
                 f"P95 ≈ {r_hig:.2f} kΩ  (HRS参考)",
                 color="#FFB74D", fontsize=7.5)
        if r_low > 0:
            ratio = r_hig / r_low
            ax2.text(0.97, 0.05,
                     f"P95/P5 ≈ {ratio:.1f}×",
                     transform=ax2.transAxes,
                     color="#CE93D8", fontsize=9.5, ha="right",
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="#1A1D27", ec="#9C27B0", alpha=0.85))

    ax2.set_yscale("log")
    ax2.set_xlabel("Sample Index",     color="#CCCCCC", fontsize=10)
    ax2.set_ylabel("Resistance (kΩ)",  color="#CCCCCC", fontsize=10)
    ax2.set_title("Resistance vs. Sample Index",
                  color="#EEEEEE", fontsize=11, pad=8)
    ax2.legend(facecolor="#1A1D27", edgecolor="#555",
               labelcolor="#CCCCCC", fontsize=8)
    ax2.grid(True, color="#2A2D37", lw=0.5, which="both", zorder=0)

    # ── 总标题（元数据摘要）───────────────────────────
    ts_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dur_str = f"{t[-1]:.2f} s" if len(t) > 0 else "—"
    fig.suptitle(
        f"Buffer: {data['buf_name']}   |   {n} pts   |   "
        f"Duration: {dur_str}   |   {ts_str}",
        color="#DDDDDD", fontsize=9, y=1.01
    )

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PNG] 已保存: {img_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Keithley 2450 Buffer 读出")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS)
    parser.add_argument("--buffer",  default=DEFAULT_BUFFER,
                        help="Buffer 名称，默认 defbuffer1")
    parser.add_argument("--output",  default=OUTPUT_PREFIX,
                        help="输出文件前缀（不含扩展名），默认 buffer_data")
    args = parser.parse_args()

    # 时间戳后缀，避免覆盖
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{args.output}_{ts}.csv"
    img_path = f"{args.output}_{ts}.png"

    inst = connect(args.address)
    data = read_buffer(inst, args.buffer)
    inst.close()

    resistance = compute_resistance(data["voltages"], data["currents"])

    # 终端打印摘要
    print(f"\n  点数      : {data['n_points']}")
    print(f"  电压范围  : {data['voltages'].min():.4f} ~ {data['voltages'].max():.4f} V")
    fin = resistance[np.isfinite(resistance)]
    if len(fin):
        print(f"  电阻范围  : {fin.min()/1e3:.3f} ~ {fin.max()/1e3:.3f} kΩ")
        print(f"  电流峰值  : {np.abs(data['currents']).max()*1e6:.3f} µA")

    save_csv(data, resistance, csv_path)
    plot_and_save(data, resistance, img_path)


if __name__ == "__main__":
    main()