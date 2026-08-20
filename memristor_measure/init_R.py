"""
measure_R_initial.py
====================
Phase-0 初始电阻分布测量脚本
器件栈: Ni / [TiOx/AlOx]x3 / Ti

仪器: Keithley 2450 (SCPI 模式, PyVISA)
功能: 逐个器件读取初始电阻, 实时打印结果, 测完后输出统计摘要

依赖:
    pip install pyvisa pyvisa-py numpy

用法:
    python measure_R_initial.py
    python measure_R_initial.py --address "TCPIP::192.168.1.10::inst0::INSTR"
    python measure_R_initial.py --address "GPIB0::24::INSTR" --n_devices 20
"""

import argparse
import time
import sys
import numpy as np

# ─────────────────────────────────────────────
# 可调参数区（根据实际情况修改这里）
# ─────────────────────────────────────────────
DEFAULT_VISA_ADDRESS = "USB0::0x05E6::0x2450::04437634::INSTR"  # USB 默认地址
V_READ        = 0.1       # 读取电压 (V) —— 小信号, 远低于成形电压
I_CC          = 50e-6      # 顺应电流 (A) —— 保护上限, 避免意外成形
N_SAMPLES     = 10        # 每个器件采样次数, 取均值
SETTLE_TIME   = 0.2       # 施加 V_READ 后的稳定等待时间 (s)
INTER_DEV_PAUSE = 1.0     # 切换器件之间的暂停时间 (s), 留给手动换接

# 筛除标准
R_SHORT_THRESH = 1e3      # 低于此值判为短路 (Ω)
R_OPEN_THRESH  = 1e8      # 高于此值判为断路 (Ω)

NPLC = 1.0                # 积分时间 (Number of Power Line Cycles), 1.0 = 16.7 ms @60Hz
# ─────────────────────────────────────────────


def connect(address: str):
    """建立 VISA 连接, 返回仪器资源对象"""
    try:
        import pyvisa
    except ImportError:
        print("[ERROR] 未找到 pyvisa 库, 请执行: pip install pyvisa pyvisa-py")
        sys.exit(1)

    rm = pyvisa.ResourceManager()
    print(f"[连接] 正在连接: {address}")
    try:
        inst = rm.open_resource(address)
    except Exception as e:
        print(f"[ERROR] 无法连接仪器: {e}")
        print("已发现的 VISA 资源:")
        for r in rm.list_resources():
            print(f"  {r}")
        sys.exit(1)

    inst.timeout = 10000  # ms
    idn = inst.query("*IDN?").strip()
    print(f"[连接] 成功 → {idn}\n")
    return inst


def init_smu(inst, v_read: float, i_cc: float, nplc: float):
    """配置 2450 为小信号电压源 + 电流测量模式"""
    cmds = [
        "*RST",                                  # 复位
        "*CLS",                                  # 清除状态寄存器
        ":SOUR:FUNC VOLT",                       # 源: 电压
        f":SOUR:VOLT:LEV {v_read:.4f}",          # 输出电压 = V_READ
        ":SOUR:VOLT:RANG:AUTO ON",               # 自动量程
        f":SOUR:VOLT:ILIM {i_cc:.6e}",           # 顺应电流保护
        ":SENS:FUNC 'CURR'",                     # 测量: 电流
        ":SENS:CURR:RANG:AUTO ON",               # 电流自动量程
        f":SENS:CURR:NPLC {nplc:.2f}",           # 积分时间
        ":SENS:CURR:AZER:ONCE",                  # 自动调零一次
        ":OUTP OFF",                             # 先关输出
    ]
    for cmd in cmds:
        inst.write(cmd)
    time.sleep(0.5)  # 等待复位完成


def measure_resistance(inst, n_samples: int, settle_time: float) -> dict:
    """
    开启输出 → 等待稳定 → 采样 N 次 → 关闭输出
    返回字典: {R_mean, R_std, I_mean, V_applied, raw_I}
    """
    inst.write(":OUTP ON")
    time.sleep(settle_time)

    currents = []
    for _ in range(n_samples):
        val = inst.query(":READ?").strip()
        # 2450 :READ? 返回 "voltage,current,resistance,timestamp,status"
        parts = val.split(",")
        try:
            i = float(parts[1])
        except (IndexError, ValueError):
            # 备用: 直接解析为单值电流
            try:
                i = float(parts[0])
            except ValueError:
                i = float("nan")
        currents.append(i)
        time.sleep(0.05)

    inst.write(":OUTP OFF")

    currents = np.array(currents)
    i_mean = np.nanmean(currents)

    # 计算电阻 R = V / I (处理零电流保护)
    v_applied = V_READ
    if abs(i_mean) < 1e-14:
        r_mean = float("inf")
    else:
        r_mean = abs(v_applied / i_mean)

    r_vals = np.where(np.abs(currents) < 1e-14, np.inf, np.abs(v_applied / currents))
    r_std = np.nanstd(r_vals[np.isfinite(r_vals)])

    return {
        "R_mean": r_mean,
        "R_std":  r_std,
        "I_mean": i_mean,
        "V_applied": v_applied,
        "raw_I": currents.tolist(),
    }


def classify(r: float) -> str:
    """器件状态分类"""
    if r < R_SHORT_THRESH:
        return "SHORT ⚠"
    if r > R_OPEN_THRESH:
        return "OPEN  ⚠"
    return "OK    ✓"


def format_resistance(r: float) -> str:
    """自适应单位格式化"""
    if not np.isfinite(r):
        return "∞ Ω (OPEN)"
    if r >= 1e9:
        return f"{r/1e9:.3f} GΩ"
    if r >= 1e6:
        return f"{r/1e6:.3f} MΩ"
    if r >= 1e3:
        return f"{r/1e3:.3f} kΩ"
    return f"{r:.2f} Ω"


def print_header():
    print("=" * 65)
    print(f"  {'器件':<8} {'R_mean':>14} {'R_std':>12} {'I_mean':>14} {'状态':<10}")
    print("-" * 65)


def print_result(dev_id: int, res: dict, status: str):
    r_str = format_resistance(res["R_mean"])
    r_std_str = format_resistance(res["R_std"])
    i_str = f"{res['I_mean']*1e6:.3f} µA"
    print(f"  DEV-{dev_id:<4} {r_str:>14} {r_std_str:>12} {i_str:>14}   {status}")


def print_summary(results: list):
    print("\n" + "=" * 65)
    print("  测量完成 — 统计摘要")
    print("=" * 65)

    r_all = [r["R_mean"] for r in results if np.isfinite(r["R_mean"])]
    if not r_all:
        print("  无有效数据。")
        return

    r_arr = np.array(r_all)
    n_ok     = sum(1 for r in results if classify(r["R_mean"]) == "OK    ✓")
    n_short  = sum(1 for r in results if classify(r["R_mean"]) == "SHORT ⚠")
    n_open   = sum(1 for r in results if classify(r["R_mean"]) == "OPEN  ⚠")

    print(f"  器件总数    : {len(results)}")
    print(f"  有效 (OK)   : {n_ok}")
    print(f"  短路 (SHORT): {n_short}  (R < {format_resistance(R_SHORT_THRESH)})")
    print(f"  断路 (OPEN) : {n_open}  (R > {format_resistance(R_OPEN_THRESH)})")
    print()
    print(f"  R 均值      : {format_resistance(np.mean(r_arr))}")
    print(f"  R 中值      : {format_resistance(np.median(r_arr))}")
    print(f"  R 最小值    : {format_resistance(np.min(r_arr))}")
    print(f"  R 最大值    : {format_resistance(np.max(r_arr))}")
    print(f"  R 标准差    : {format_resistance(np.std(r_arr))}")

    # 分布直方图（ASCII 版，无需 matplotlib）
    if len(r_arr) >= 3:
        print()
        print("  对数电阻分布 (log10 R / Ω):")
        log_r = np.log10(r_arr)
        bins = np.linspace(log_r.min(), log_r.max(), 8)
        counts, edges = np.histogram(log_r, bins=bins)
        max_count = max(counts) if max(counts) > 0 else 1
        bar_width = 30
        for i in range(len(counts)):
            lo = 10 ** edges[i]
            hi = 10 ** edges[i+1]
            bar = "█" * int(counts[i] / max_count * bar_width)
            print(f"  {format_resistance(lo):>10} – {format_resistance(hi):<10} │{bar:<{bar_width}}│ {counts[i]}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Keithley 2450 初始电阻测量")
    parser.add_argument("--address",   default=DEFAULT_VISA_ADDRESS, help="VISA 资源地址")
    parser.add_argument("--n_devices", type=int, default=None,       help="器件数量 (不填则手动确认每颗)")
    parser.add_argument("--output",    default=None,                  help="结果输出 CSV 文件路径 (可选)")
    args = parser.parse_args()

    inst = connect(args.address)
    init_smu(inst, V_READ, I_CC, NPLC)

    print(f"  测量参数:")
    print(f"    V_read    = {V_READ} V")
    print(f"    I_cc      = {I_CC*1e3:.1f} mA (保护上限)")
    print(f"    每器件采样 = {N_SAMPLES} 次")
    print(f"    稳定等待  = {SETTLE_TIME} s")
    print()

    results = []
    dev_id  = 1

    print_header()

    try:
        while True:
            # 确定本次是否继续
            if args.n_devices is not None:
                if dev_id > args.n_devices:
                    break
                # 自动模式: 提示用户换接线
                input(f"\n  → 请接好 DEV-{dev_id}，然后按 Enter 开始测量...")
            else:
                # 手动模式: 每次询问是否继续
                ans = input(f"\n  → 请接好 DEV-{dev_id}，按 Enter 测量 / 输入 q 退出: ").strip().lower()
                if ans == "q":
                    break

            res = measure_resistance(inst, N_SAMPLES, SETTLE_TIME)
            status = classify(res["R_mean"])
            print_result(dev_id, res, status)
            results.append({"dev_id": dev_id, **res, "status": status})

            time.sleep(INTER_DEV_PAUSE)
            dev_id += 1

    except KeyboardInterrupt:
        print("\n\n  [中断] 用户手动停止。")

    finally:
        # 确保仪器输出关闭, 电压归零
        try:
            inst.write(":OUTP OFF")
            inst.write(":SOUR:VOLT:LEV 0")
        except Exception:
            pass
        inst.close()

    if results:
        print_summary(results)

        # 可选: 保存 CSV
        if args.output:
            import csv
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["dev_id", "R_mean", "R_std", "I_mean", "V_applied", "status"])
                writer.writeheader()
                for r in results:
                    row = {k: r[k] for k in ["dev_id", "R_mean", "R_std", "I_mean", "V_applied", "status"]}
                    writer.writerow(row)
            print(f"\n  结果已保存至: {args.output}")


if __name__ == "__main__":
    main()