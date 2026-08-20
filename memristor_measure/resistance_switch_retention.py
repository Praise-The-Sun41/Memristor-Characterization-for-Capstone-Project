"""
Resistance switching and retention test for Keithley 2450.

Purpose
-------
Check whether a fabricated memristor device can be switched between resistance
states and whether the state is retained after SET/RESET write steps.

Measurement flow
----------------
For each cycle:
  1. Read baseline resistance at V_READ.
  2. Apply one SET write step at V_SET for T_SET.
  3. Read resistance repeatedly at V_READ for SET retention.
  4. Apply one RESET write step at V_RESET for T_RESET.
  5. Read resistance repeatedly at V_READ for RESET retention.

All samples are saved to CSV and plotted as resistance/conductance over time.

Example
-------
python memristor_measure/resistance_switch_retention.py --dev DEV-1-1
python memristor_measure/resistance_switch_retention.py --v-set 2 --v-reset -2 --cycles 3
"""

import argparse
import csv
import json
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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Switch_Retention")

V_READ = 0.1
V_SET = 2.0
V_RESET = -2.0
T_SET = 0.5
T_RESET = 0.5
READ_INTERVAL = 0.2
RETENTION_TIME = 5.0
BASELINE_TIME = 1.0
I_CC = 50e-6
NPLC = 0.5
N_READ_AVG = 3
READ_SETTLE = 0.02


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
    except Exception as exc:
        print(f"[ERROR] Connection failed: {exc}")
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


def init_smu(inst, i_cc: float, nplc: float):
    cmds = [
        "*RST",
        "*CLS",
        ":SOUR:FUNC VOLT",
        ":SOUR:VOLT:RANG:AUTO 1",
        f":SOUR:VOLT:ILIM {i_cc:.6e}",
        ":SOUR:VOLT:LEV 0",
        ':SENS:FUNC "CURR"',
        ":SENS:CURR:RANG:AUTO 1",
        f":SENS:CURR:NPLC {nplc:.3f}",
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
        print(f"[Init] Voltage source mode ready, Icc={i_cc:.6e} A, NPLC={nplc}")


def read_current_once(inst) -> float:
    raw = inst.query(":READ?").strip()
    parts = raw.split(",")
    return float(parts[1]) if len(parts) >= 2 else float(parts[0])


def resistance_from(voltage: float, current: float) -> float:
    if not np.isfinite(current) or abs(current) <= 1e-14:
        return float("inf")
    return abs(voltage / current)


def fmt_resistance(resistance: float) -> str:
    if not np.isfinite(resistance):
        return "inf Ohm"
    if resistance >= 1e9:
        return f"{resistance / 1e9:.3f} GOhm"
    if resistance >= 1e6:
        return f"{resistance / 1e6:.3f} MOhm"
    if resistance >= 1e3:
        return f"{resistance / 1e3:.3f} kOhm"
    return f"{resistance:.3f} Ohm"


def read_resistance(inst, v_read: float, n_avg: int, settle: float):
    inst.write(f":SOUR:VOLT:LEV {v_read:.6f}")
    inst.write(":OUTP ON")
    time.sleep(settle)

    currents = []
    for _ in range(n_avg):
        try:
            currents.append(read_current_once(inst))
        except Exception as exc:
            print(f"[Read warning] {exc}")
        time.sleep(0.002)

    inst.write(":SOUR:VOLT:LEV 0")

    if not currents:
        return float("nan"), float("inf"), 0.0

    current = float(np.mean(currents))
    resistance = resistance_from(v_read, current)
    conductance = 0.0 if not np.isfinite(resistance) or resistance <= 0 else 1.0 / resistance
    return current, resistance, conductance


def apply_write_step(inst, voltage: float, duration: float):
    inst.write(f":SOUR:VOLT:LEV {voltage:.6f}")
    inst.write(":OUTP ON")
    time.sleep(duration)
    inst.write(":SOUR:VOLT:LEV 0")


def append_sample(records, cycle, phase, elapsed, v_write, v_read, current, resistance, conductance):
    records.append(
        {
            "index": len(records),
            "cycle": cycle,
            "phase": phase,
            "time_s": elapsed,
            "write_voltage_V": v_write,
            "read_voltage_V": v_read,
            "current_A": current,
            "resistance_Ohm": resistance,
            "conductance_S": conductance,
        }
    )


def sample_for_duration(inst, records, cycle, phase, duration, interval, v_read, n_avg, settle, t_zero, v_write=0.0):
    end_time = time.time() + max(duration, 0.0)
    next_sample = time.time()
    while True:
        now = time.time()
        if now >= end_time and len(records) > 0 and records[-1]["phase"] == phase:
            break
        if now < next_sample:
            time.sleep(min(0.01, next_sample - now))
            continue

        current, resistance, conductance = read_resistance(inst, v_read, n_avg, settle)
        elapsed = time.time() - t_zero
        append_sample(records, cycle, phase, elapsed, v_write, v_read, current, resistance, conductance)
        print(
            f"  {len(records)-1:5d}  cycle={cycle:02d}  {phase:>15}  "
            f"t={elapsed:8.3f}s  R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS"
        )
        next_sample += interval
        if duration <= 0:
            break


def save_csv(records, csv_path: str):
    columns = [
        "index",
        "cycle",
        "phase",
        "time_s",
        "write_voltage_V",
        "read_voltage_V",
        "current_A",
        "resistance_Ohm",
        "conductance_S",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            out = dict(row)
            for key in ("time_s", "write_voltage_V", "read_voltage_V", "current_A", "resistance_Ohm", "conductance_S"):
                value = out[key]
                if isinstance(value, float):
                    out[key] = f"{value:.8e}" if np.isfinite(value) else "inf"
            writer.writerow(out)
    print(f"[CSV] Saved -> {csv_path}")


def save_metadata(args, meta_path: str):
    payload = vars(args).copy()
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[JSON] Saved -> {meta_path}")


def plot_records(records, img_path: str, title: str):
    if not records:
        return

    times = np.array([r["time_s"] for r in records], dtype=float)
    resistances = np.array([r["resistance_Ohm"] for r in records], dtype=float)
    conductances = np.array([r["conductance_S"] for r in records], dtype=float)
    phases = [r["phase"] for r in records]

    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor("#0F1117")
    for ax in (ax_r, ax_g):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.grid(True, color="#2A2D37", lw=0.5, which="both")

    colors = {
        "baseline": "#B0BEC5",
        "after_set": "#4DB6AC",
        "set_retention": "#26A69A",
        "after_reset": "#FFB74D",
        "reset_retention": "#FF9800",
    }
    for phase in sorted(set(phases)):
        mask = np.array([p == phase for p in phases])
        finite_r = np.isfinite(resistances) & mask
        ax_r.plot(times[finite_r], resistances[finite_r], "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)
        ax_g.plot(times[mask], conductances[mask] * 1e6, "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)

    if np.any(np.isfinite(resistances)):
        ax_r.set_yscale("log")
    ax_r.set_ylabel("Resistance (Ohm)", color="#CCCCCC")
    ax_r.set_title("Resistance retention", color="#EEEEEE", fontsize=11)
    ax_r.legend(facecolor="#1A1D27", edgecolor="#444", labelcolor="#CCCCCC", fontsize=8)

    ax_g.set_xlabel("Time (s)", color="#CCCCCC")
    ax_g.set_ylabel("Conductance (uS)", color="#CCCCCC")
    ax_g.set_title("Conductance change", color="#EEEEEE", fontsize=11)

    fig.suptitle(title, color="#DDDDDD", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def make_run_dir(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.dev}_SET{args.v_set:+.2f}_RESET{args.v_reset:+.2f}_"
        f"read{args.v_read:+.2f}_cyc{args.cycles}_{ts}"
    )
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts


def main():
    parser = argparse.ArgumentParser(description="Memristor resistance switching and retention test")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS, help="VISA resource address")
    parser.add_argument("--dev", default="DEV", help="Device label for output folder naming")
    parser.add_argument("--v-read", type=float, default=V_READ, help="Non-destructive read voltage in V")
    parser.add_argument("--v-set", type=float, default=V_SET, help="SET write voltage in V")
    parser.add_argument("--v-reset", type=float, default=V_RESET, help="RESET write voltage in V")
    parser.add_argument("--t-set", type=float, default=T_SET, help="SET write duration in seconds")
    parser.add_argument("--t-reset", type=float, default=T_RESET, help="RESET write duration in seconds")
    parser.add_argument("--baseline-time", type=float, default=BASELINE_TIME, help="Baseline read duration in seconds")
    parser.add_argument("--retention-time", type=float, default=RETENTION_TIME, help="Retention read duration after each write")
    parser.add_argument("--interval", type=float, default=READ_INTERVAL, help="Read sample interval in seconds")
    parser.add_argument("--icc", type=float, default=I_CC, help="Source current compliance in A")
    parser.add_argument("--nplc", type=float, default=NPLC, help="Current measurement NPLC")
    parser.add_argument("--n-read-avg", type=int, default=N_READ_AVG, help="Number of read samples to average")
    parser.add_argument("--read-settle", type=float, default=READ_SETTLE, help="Settling time before each read in seconds")
    parser.add_argument("--cycles", type=int, default=1, help="Number of SET/RESET cycles")
    parser.add_argument("--no-plot", action="store_true", help="Save CSV only")
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    if args.icc <= 0:
        raise ValueError("--icc must be > 0")
    if args.cycles <= 0:
        raise ValueError("--cycles must be > 0")
    if args.n_read_avg <= 0:
        raise ValueError("--n-read-avg must be > 0")

    run_dir, ts = make_run_dir(args)
    csv_path = os.path.join(run_dir, f"switch_retention_{ts}.csv")
    img_path = os.path.join(run_dir, f"switch_retention_{ts}.png")
    meta_path = os.path.join(run_dir, f"switch_retention_{ts}.json")

    print("=" * 68)
    print("  Resistance Switching and Retention Test")
    print("=" * 68)
    print(f"  Device       : {args.dev}")
    print(f"  SET          : {args.v_set:+.4g} V for {args.t_set:.3f} s")
    print(f"  RESET        : {args.v_reset:+.4g} V for {args.t_reset:.3f} s")
    print(f"  Read         : {args.v_read:+.4g} V, avg={args.n_read_avg}, interval={args.interval:.3f} s")
    print(f"  Retention    : {args.retention_time:.3f} s after each write")
    print(f"  Cycles       : {args.cycles}")
    print(f"  Compliance   : {args.icc:.3e} A")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    ans = input("\nConfirm start? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    records = []
    inst = connect(args.address)
    t_zero = time.time()
    try:
        init_smu(inst, args.icc, args.nplc)
        sample_for_duration(
            inst, records, 0, "baseline", args.baseline_time, args.interval,
            args.v_read, args.n_read_avg, args.read_settle, t_zero
        )

        for cycle in range(1, args.cycles + 1):
            print(f"\n[Cycle {cycle}] SET write {args.v_set:+.4g} V, {args.t_set:.3f} s")
            apply_write_step(inst, args.v_set, args.t_set)
            current, resistance, conductance = read_resistance(inst, args.v_read, args.n_read_avg, args.read_settle)
            append_sample(records, cycle, "after_set", time.time() - t_zero, args.v_set, args.v_read, current, resistance, conductance)
            print(f"  after SET   R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS")
            sample_for_duration(
                inst, records, cycle, "set_retention", args.retention_time, args.interval,
                args.v_read, args.n_read_avg, args.read_settle, t_zero, args.v_set
            )

            print(f"\n[Cycle {cycle}] RESET write {args.v_reset:+.4g} V, {args.t_reset:.3f} s")
            apply_write_step(inst, args.v_reset, args.t_reset)
            current, resistance, conductance = read_resistance(inst, args.v_read, args.n_read_avg, args.read_settle)
            append_sample(records, cycle, "after_reset", time.time() - t_zero, args.v_reset, args.v_read, current, resistance, conductance)
            print(f"  after RESET R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS")
            sample_for_duration(
                inst, records, cycle, "reset_retention", args.retention_time, args.interval,
                args.v_read, args.n_read_avg, args.read_settle, t_zero, args.v_reset
            )
    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C received, saving collected data.")
    finally:
        try:
            inst.write(":SOUR:VOLT:LEV 0")
            inst.write(":OUTP OFF")
            print("[Safe] Output OFF, voltage returned to 0 V.")
        finally:
            inst.close()

    save_csv(records, csv_path)
    save_metadata(args, meta_path)

    finite_r = np.array([r["resistance_Ohm"] for r in records if np.isfinite(r["resistance_Ohm"])], dtype=float)
    if len(finite_r):
        print(f"[Summary] R range: {fmt_resistance(float(np.min(finite_r)))} to {fmt_resistance(float(np.max(finite_r)))}")
        if np.min(finite_r) > 0:
            print(f"[Summary] Rmax/Rmin: {float(np.max(finite_r) / np.min(finite_r)):.3f}x")

    if not args.no_plot:
        title = f"{args.dev} switching retention: SET {args.v_set:+.2f} V, RESET {args.v_reset:+.2f} V"
        plot_records(records, img_path, title)


if __name__ == "__main__":
    main()

