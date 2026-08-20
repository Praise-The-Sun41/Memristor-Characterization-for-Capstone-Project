"""
Fixed voltage current monitor for Keithley 2450.

This script follows the same direct SCPI/PyVISA style as IV_Sweep.py:
  - source a fixed voltage
  - continuously read current
  - stream data to CSV while running
  - turn output off and return to 0 V on exit
  - save a time-current / time-resistance plot after the run

Usage examples:
    python Fixed_V_Read.py
    python Fixed_V_Read.py --voltage 0.1 --interval 0.5 --duration 300
    python Fixed_V_Read.py --voltage -0.2 --icc 50e-6 --dev DEV-2-2
    python Fixed_V_Read.py --voltage 5 --baseline-voltage 0.1 --interval 0.01 --pre-time 1 --settle 0.5 --duration 5

Set --duration 0 to run until Ctrl+C.
The baseline is also sampled repeatedly: baseline sample count ~= --pre-time / --interval.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import matplotlib
import platform

_fonts = {"Windows": "Microsoft YaHei", "Darwin": "PingFang SC", "Linux": "WenQuanYi Micro Hei"}
matplotlib.rcParams["font.family"] = _fonts.get(platform.system(), "DejaVu Sans")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_VISA_ADDRESS = "USB0::0x05E6::0x2450::04437634::INSTR"

V_READ = 0.1
I_CC = 50e-6
NPLC = 0.5
SAMPLE_INTERVAL = 0.1
DURATION = 5.0
SETTLE_TIME = 1.0
PRE_TIME = 1.0
BASELINE_VOLTAGE = 0.1

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fixed_V_Read")


def connect(address: str):
    try:
        import pyvisa
    except ImportError:
        print("[ERROR] Missing dependency: pip install pyvisa pyvisa-py")
        sys.exit(1)

    rm = pyvisa.ResourceManager()
    print(f"[Connect] {address}")
    try:
        inst = rm.open_resource(address)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("Available VISA resources:")
        for resource in rm.list_resources():
            print(f"  {resource}")
        sys.exit(1)

    inst.write_termination = "\n"
    inst.read_termination = "\n"
    inst.timeout = 15000

    try:
        inst.clear()
        time.sleep(0.3)
    except Exception:
        pass

    inst.write("*CLS")
    time.sleep(0.1)

    idn = inst.query("*IDN?").strip()
    print(f"[Connect] OK -> {idn}\n")
    return inst


def init_smu(inst, voltage: float, i_cc: float, nplc: float):
    cmds = [
        "*RST",
        "*CLS",
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
    for cmd in cmds:
        inst.write(cmd)
    time.sleep(0.3)

    err = inst.query(":SYST:ERR?").strip()
    if not err.startswith("0"):
        print(f"[Warning] Init instrument error: {err}")
    else:
        print(f"[Init] Fixed voltage mode ready, V={voltage:.6g} V, Icc={i_cc:.6e} A")


def read_current(inst) -> float:
    raw = inst.query(":READ?").strip()
    parts = raw.split(",")
    return float(parts[1]) if len(parts) >= 2 else float(parts[0])


def resistance_from(voltage: float, current: float) -> float:
    if not np.isfinite(current) or abs(current) <= 1e-14:
        return float("inf")
    return abs(voltage / current)


def fmt_current(current: float) -> str:
    if not np.isfinite(current):
        return "nan"
    if abs(current) >= 1e-3:
        return f"{current * 1e3:.4f} mA"
    if abs(current) >= 1e-6:
        return f"{current * 1e6:.4f} uA"
    if abs(current) >= 1e-9:
        return f"{current * 1e9:.4f} nA"
    return f"{current * 1e12:.4f} pA"


def fmt_resistance(resistance: float) -> str:
    if not np.isfinite(resistance):
        return "inf Ohm"
    if resistance >= 1e9:
        return f"{resistance / 1e9:.4f} GOhm"
    if resistance >= 1e6:
        return f"{resistance / 1e6:.4f} MOhm"
    if resistance >= 1e3:
        return f"{resistance / 1e3:.4f} kOhm"
    return f"{resistance:.4f} Ohm"


def run_monitor(
    inst,
    voltage: float,
    baseline_voltage: float,
    interval: float,
    duration: float,
    settle: float,
    pre_time: float,
    csv_path: str,
):
    times = []
    currents = []
    resistances = []
    voltages = []

    print(f"[Run] Output ON at baseline voltage {baseline_voltage:.6g} V")
    inst.write(f":SOUR:VOLT:LEV {baseline_voltage:.6f}")
    inst.write(":OUTP ON")

    print(f"[Run] Set voltage to {voltage:.6g} V and start sampling immediately")

    print()
    print(f"  {'Index':>7}  {'Phase':>9}  {'Time (s)':>10}  {'V (V)':>9}  {'I':>14}  {'R':>14}")
    print("  " + "-" * 74)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "phase",
            "time_s",
            "voltage_V",
            "current_A",
            "current_uA",
            "resistance_Ohm",
        ])

        index = 0

        def sample(phase: str, t_zero: float, applied_voltage: float):
            nonlocal index
            try:
                current = read_current(inst)
            except Exception as e:
                print(f"  [Read warning #{index}] {e}")
                current = float("nan")

            elapsed = time.time() - t_zero
            resistance = resistance_from(applied_voltage, current)

            times.append(elapsed)
            voltages.append(applied_voltage)
            currents.append(current)
            resistances.append(resistance)

            writer.writerow([
                index,
                phase,
                f"{elapsed:.6f}",
                f"{applied_voltage:.6f}",
                f"{current:.6e}",
                f"{current * 1e6:.6f}" if np.isfinite(current) else "nan",
                f"{resistance:.6e}" if np.isfinite(resistance) else "inf",
            ])
            f.flush()

            print(
                f"  {index:7d}  {phase:>9}  {elapsed:10.3f}  {applied_voltage:9.4f}  "
                f"{fmt_current(current):>14}  {fmt_resistance(resistance):>14}"
            )
            index += 1

        if pre_time > 0:
            t_pre_start = time.time()
            next_sample = t_pre_start
            est_baseline_points = max(1, int(np.ceil(pre_time / interval)))
            print(
                f"[Run] Recording {baseline_voltage:.6g} V baseline repeatedly "
                f"for {pre_time:.3f} s, about {est_baseline_points} samples"
            )
            while True:
                now = time.time()
                if now - t_pre_start >= pre_time:
                    break
                if now < next_sample:
                    time.sleep(min(0.01, next_sample - now))
                    continue
                sample("baseline", t_pre_start + pre_time, baseline_voltage)
                next_sample += interval

        inst.write(f":SOUR:VOLT:LEV {voltage:.6f}")
        t_start = time.time()
        next_sample = t_start
        total_duration = duration + max(settle, 0.0) if duration > 0 else 0

        while True:
            now = time.time()
            elapsed = now - t_start
            if total_duration > 0 and elapsed >= total_duration:
                break

            if now < next_sample:
                time.sleep(min(0.01, next_sample - now))
                continue

            phase = "settling" if elapsed < settle else "read"
            sample(phase, t_start, voltage)
            next_sample += interval

    return np.array(times), np.array(voltages), np.array(currents), np.array(resistances)


def plot_and_save(times, voltages, currents, resistances, voltage: float, baseline_voltage: float, settle: float, img_path: str):
    if len(times) == 0:
        print("[PNG] No samples collected, skip plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor("#0F1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.grid(True, color="#2A2D37", lw=0.5, which="both")

    ax1.plot(times, currents * 1e6, color="#4DB6AC", lw=1.6)
    ax1.scatter(times, currents * 1e6, color="#80CBC4", s=10, alpha=0.75)
    ax1.axvline(0, color="#777", lw=0.8, linestyle="--")
    if settle > 0:
        ax1.axvline(settle, color="#777", lw=0.8, linestyle=":")
    ax1.axhline(0, color="#555", lw=0.7)
    ax1.set_ylabel("Current (uA)", color="#CCCCCC")
    ax1.set_title("Current vs. Time", color="#EEEEEE", fontsize=11, pad=8)

    finite_r = np.isfinite(resistances)
    if np.any(finite_r):
        ax2.plot(times[finite_r], resistances[finite_r] / 1e3, color="#FFB74D", lw=1.4)
        ax2.scatter(times[finite_r], resistances[finite_r] / 1e3, color="#FFD54F", s=10, alpha=0.75)
        ax2.set_yscale("log")
    ax2.axvline(0, color="#777", lw=0.8, linestyle="--")
    if settle > 0:
        ax2.axvline(settle, color="#777", lw=0.8, linestyle=":")
    ax2.set_xlabel("Time (s)", color="#CCCCCC")
    ax2.set_ylabel("Resistance (kOhm)", color="#CCCCCC")
    ax2.set_title("Resistance vs. Time", color="#EEEEEE", fontsize=11, pad=8)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actual_v = voltages[np.isfinite(voltages)]
    v_label = f"{np.nanmin(actual_v):.3g}~{np.nanmax(actual_v):.3g} V" if len(actual_v) else f"{voltage:.6g} V"
    fig.suptitle(
        f"Fixed Voltage Read   baseline={baseline_voltage:.6g} V, read={voltage:.6g} V   "
        f"V={v_label}   Samples={len(times)}   {ts}",
        color="#DDDDDD", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def make_run_dir(dev: str, voltage: float, i_cc: float, nplc: float, interval: float, duration: float):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dur_label = "untilstop" if duration <= 0 else f"{int(duration)}s"
    run_name = (
        f"{dev}_"
        f"V{voltage:+.3f}_"
        f"dt{int(interval * 1000)}ms_"
        f"dur{dur_label}_"
        f"Icc{int(i_cc * 1e6)}uA_"
        f"NPLC{nplc}_"
        f"{ts_str}"
    )
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts_str


def main():
    parser = argparse.ArgumentParser(description="Keithley 2450 fixed-voltage continuous current read")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS, help="VISA resource address")
    parser.add_argument("--dev", default="DEV", help="Device label for output folder naming")
    parser.add_argument("--voltage", type=float, default=V_READ, help="Fixed source voltage in V")
    parser.add_argument("--baseline-voltage", type=float, default=BASELINE_VOLTAGE, help="Small read voltage used during --pre-time baseline")
    parser.add_argument("--icc", type=float, default=I_CC, help="Source current compliance in A")
    parser.add_argument("--nplc", type=float, default=NPLC, help="Current measurement NPLC")
    parser.add_argument("--interval", type=float, default=SAMPLE_INTERVAL, help="Sample interval in seconds")
    parser.add_argument("--duration", type=float, default=DURATION, help="Run duration in seconds; 0 means until Ctrl+C")
    parser.add_argument("--settle", type=float, default=SETTLE_TIME, help="Settling window in seconds; data is recorded during this window")
    parser.add_argument("--pre-time", type=float, default=PRE_TIME, help="Baseline sampling time before applying target voltage; sampled repeatedly at --interval")
    parser.add_argument("--no-plot", action="store_true", help="Only save CSV, skip PNG plot")
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    if args.icc <= 0:
        raise ValueError("--icc must be > 0")
    if args.nplc <= 0:
        raise ValueError("--nplc must be > 0")
    if args.settle < 0:
        raise ValueError("--settle must be >= 0")
    if args.pre_time < 0:
        raise ValueError("--pre-time must be >= 0")

    run_dir, ts_str = make_run_dir(args.dev, args.voltage, args.icc, args.nplc, args.interval, args.duration)
    csv_path = os.path.join(run_dir, f"fixed_v_read_{ts_str}.csv")
    img_path = os.path.join(run_dir, f"fixed_v_read_{ts_str}.png")

    duration_label = "until Ctrl+C" if args.duration <= 0 else f"{args.duration:.1f} s"
    baseline_points = int(np.ceil(args.pre_time / args.interval)) if args.pre_time > 0 else 0
    print("=" * 58)
    print("  Fixed Voltage Current Monitor")
    print("=" * 58)
    print(f"  Device label    = {args.dev}")
    print(f"  Voltage         = {args.voltage:.6g} V")
    print(f"  Baseline V      = {args.baseline_voltage:.6g} V")
    print(f"  I compliance    = {args.icc:.6e} A")
    print(f"  NPLC            = {args.nplc}")
    print(f"  Sample interval = {args.interval:.3f} s")
    print(f"  Duration        = {duration_label}")
    print(f"  Settling window = {args.settle:.3f} s  (recorded, not skipped)")
    print(f"  Baseline time   = {args.pre_time:.3f} s")
    print(f"  Baseline points = about {baseline_points}")
    print(f"  Output dir      = {run_dir}")
    print("=" * 58)

    ans = input("\nConfirm start? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    inst = connect(args.address)
    times = np.array([])
    voltages = np.array([])
    currents = np.array([])
    resistances = np.array([])

    try:
        init_smu(inst, args.voltage, args.icc, args.nplc)
        times, voltages, currents, resistances = run_monitor(
            inst,
            args.voltage,
            args.baseline_voltage,
            args.interval,
            args.duration,
            args.settle,
            args.pre_time,
            csv_path,
        )
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, stopping measurement.")
    finally:
        try:
            inst.write(":SOUR:VOLT:LEV 0")
            inst.write(":OUTP OFF")
            print("[Safe] Output OFF, voltage returned to 0 V.")
        finally:
            inst.close()

    print(f"\n[CSV] Saved -> {csv_path}")
    if len(currents):
        finite_i = currents[np.isfinite(currents)]
        finite_r = resistances[np.isfinite(resistances)]
        print(f"[Summary] Samples: {len(currents)}")
        if len(finite_i):
            print(f"[Summary] Current mean: {fmt_current(float(np.mean(finite_i)))}")
            print(f"[Summary] Current min/max: {fmt_current(float(np.min(finite_i)))} / {fmt_current(float(np.max(finite_i)))}")
        if len(finite_r):
            print(f"[Summary] Resistance median: {fmt_resistance(float(np.median(finite_r)))}")

    if not args.no_plot:
        plot_and_save(times, voltages, currents, resistances, args.voltage, args.baseline_voltage, args.settle, img_path)


if __name__ == "__main__":
    main()
