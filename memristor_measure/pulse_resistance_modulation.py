"""
Pulse resistance modulation test for Keithley 2450.

Purpose
-------
Check whether a memristor device resistance can be gradually modulated by
positive and negative pulse trains. The script applies one pulse, reads the
resistance at a small V_READ, saves the result, and repeats.

This is a millisecond-level SMU pulse proxy, not a nanosecond pulse generator.

Example
-------
python memristor_measure/pulse_resistance_modulation.py --dev DEV-1-1
python memristor_measure/pulse_resistance_modulation.py --v-pot 1.2 --v-dep -1.2 --n-pot 20 --n-dep 20
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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pulse_Modulation")

V_READ = 0.1
V_POT = 0.8
V_DEP = -0.8
T_PULSE = 0.1
T_RELAX = 0.1
N_POT = 20
N_DEP = 20
N_CYCLES = 1
I_CC = 50e-6
NPLC = 0.1
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


def apply_pulse(inst, voltage: float, width: float, relax: float):
    inst.write(f":SOUR:VOLT:LEV {voltage:.6f}")
    inst.write(":OUTP ON")
    time.sleep(width)
    inst.write(":SOUR:VOLT:LEV 0")
    time.sleep(relax)


def add_record(records, cycle, phase, pulse_index, global_pulse, elapsed, v_pulse, current, resistance, conductance):
    records.append(
        {
            "index": len(records),
            "cycle": cycle,
            "phase": phase,
            "pulse_index": pulse_index,
            "global_pulse": global_pulse,
            "time_s": elapsed,
            "pulse_voltage_V": v_pulse,
            "current_A": current,
            "resistance_Ohm": resistance,
            "conductance_S": conductance,
        }
    )


def save_csv(records, csv_path: str):
    columns = [
        "index",
        "cycle",
        "phase",
        "pulse_index",
        "global_pulse",
        "time_s",
        "pulse_voltage_V",
        "current_A",
        "resistance_Ohm",
        "conductance_S",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            out = dict(row)
            for key in ("time_s", "pulse_voltage_V", "current_A", "resistance_Ohm", "conductance_S"):
                value = out[key]
                if isinstance(value, float):
                    out[key] = f"{value:.8e}" if np.isfinite(value) else "inf"
            writer.writerow(out)
    print(f"[CSV] Saved -> {csv_path}")


def save_metadata(args, meta_path: str):
    payload = vars(args).copy()
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["note"] = "Keithley 2450 millisecond-level SMU pulse proxy."
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[JSON] Saved -> {meta_path}")


def plot_records(records, img_path: str, title: str):
    if not records:
        return

    x = np.array([r["global_pulse"] for r in records], dtype=float)
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

    colors = {"baseline": "#B0BEC5", "potentiation": "#4DB6AC", "depression": "#FFB74D"}
    for phase in sorted(set(phases)):
        mask = np.array([p == phase for p in phases])
        finite_r = mask & np.isfinite(resistances)
        ax_r.plot(x[finite_r], resistances[finite_r], "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)
        ax_g.plot(x[mask], conductances[mask] * 1e6, "o-", ms=4, lw=1.2, color=colors.get(phase, "#90CAF9"), label=phase)

    if np.any(np.isfinite(resistances)):
        ax_r.set_yscale("log")
    ax_r.set_ylabel("Resistance (Ohm)", color="#CCCCCC")
    ax_r.set_title("Resistance after each pulse", color="#EEEEEE", fontsize=11)
    ax_r.legend(facecolor="#1A1D27", edgecolor="#444", labelcolor="#CCCCCC", fontsize=8)

    ax_g.set_xlabel("Pulse number", color="#CCCCCC")
    ax_g.set_ylabel("Conductance (uS)", color="#CCCCCC")
    ax_g.set_title("Conductance modulation", color="#EEEEEE", fontsize=11)

    fig.suptitle(title, color="#DDDDDD", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PNG] Saved -> {img_path}")
    plt.show()


def make_run_dir(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.dev}_POT{args.v_pot:+.2f}_DEP{args.v_dep:+.2f}_"
        f"pulse{int(args.t_pulse * 1000)}ms_N{args.n_pot}x{args.n_dep}_cyc{args.cycles}_{ts}"
    )
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts


def run_phase(inst, args, records, cycle, phase, voltage, count, global_pulse, t_zero):
    print(f"\n[Cycle {cycle}] {phase} pulses: V={voltage:+.4g} V, width={args.t_pulse:.4g} s, count={count}")
    for idx in range(1, count + 1):
        apply_pulse(inst, voltage, args.t_pulse, args.t_relax)
        current, resistance, conductance = read_resistance(inst, args.v_read, args.n_read_avg, args.read_settle)
        global_pulse += 1
        add_record(records, cycle, phase, idx, global_pulse, time.time() - t_zero, voltage, current, resistance, conductance)
        print(
            f"  pulse={global_pulse:04d}  {phase:>13} #{idx:03d}  "
            f"R={fmt_resistance(resistance):>14}  G={conductance*1e6:9.3f} uS"
        )
    return global_pulse


def main():
    parser = argparse.ArgumentParser(description="Memristor pulse resistance modulation test")
    parser.add_argument("--address", default=DEFAULT_VISA_ADDRESS, help="VISA resource address")
    parser.add_argument("--dev", default="DEV", help="Device label for output folder naming")
    parser.add_argument("--v-read", type=float, default=V_READ, help="Non-destructive read voltage in V")
    parser.add_argument("--v-pot", type=float, default=V_POT, help="Potentiation pulse voltage in V")
    parser.add_argument("--v-dep", type=float, default=V_DEP, help="Depression pulse voltage in V")
    parser.add_argument("--t-pulse", type=float, default=T_PULSE, help="Pulse width in seconds")
    parser.add_argument("--t-relax", type=float, default=T_RELAX, help="Relaxation time after each pulse in seconds")
    parser.add_argument("--n-pot", type=int, default=N_POT, help="Number of potentiation pulses per cycle")
    parser.add_argument("--n-dep", type=int, default=N_DEP, help="Number of depression pulses per cycle")
    parser.add_argument("--cycles", type=int, default=N_CYCLES, help="Number of potentiation/depression cycles")
    parser.add_argument("--icc", type=float, default=I_CC, help="Source current compliance in A")
    parser.add_argument("--nplc", type=float, default=NPLC, help="Current measurement NPLC")
    parser.add_argument("--n-read-avg", type=int, default=N_READ_AVG, help="Number of read samples to average")
    parser.add_argument("--read-settle", type=float, default=READ_SETTLE, help="Settling time before each read in seconds")
    parser.add_argument("--no-plot", action="store_true", help="Save CSV only")
    args = parser.parse_args()

    if args.icc <= 0:
        raise ValueError("--icc must be > 0")
    if args.t_pulse <= 0:
        raise ValueError("--t-pulse must be > 0")
    if args.t_relax < 0:
        raise ValueError("--t-relax must be >= 0")
    if args.n_pot < 0 or args.n_dep < 0:
        raise ValueError("--n-pot and --n-dep must be >= 0")
    if args.cycles <= 0:
        raise ValueError("--cycles must be > 0")
    if args.n_read_avg <= 0:
        raise ValueError("--n-read-avg must be > 0")

    run_dir, ts = make_run_dir(args)
    csv_path = os.path.join(run_dir, f"pulse_modulation_{ts}.csv")
    img_path = os.path.join(run_dir, f"pulse_modulation_{ts}.png")
    meta_path = os.path.join(run_dir, f"pulse_modulation_{ts}.json")

    total_pulses = (args.n_pot + args.n_dep) * args.cycles
    est_time = total_pulses * (args.t_pulse + args.t_relax + args.read_settle + 0.01)
    print("=" * 68)
    print("  Pulse Resistance Modulation Test")
    print("=" * 68)
    print(f"  Device       : {args.dev}")
    print(f"  Potentiation : {args.v_pot:+.4g} V x {args.n_pot} pulses")
    print(f"  Depression   : {args.v_dep:+.4g} V x {args.n_dep} pulses")
    print(f"  Pulse width  : {args.t_pulse:.4g} s")
    print(f"  Relax        : {args.t_relax:.4g} s")
    print(f"  Read         : {args.v_read:+.4g} V, avg={args.n_read_avg}")
    print(f"  Cycles       : {args.cycles}")
    print(f"  Total pulses : {total_pulses}, estimated time {est_time:.1f} s")
    print(f"  Compliance   : {args.icc:.3e} A")
    print(f"  Output       : {run_dir}")
    print("=" * 68)
    print("  Note: Keithley 2450 is an SMU; this is a millisecond pulse proxy.")
    print("=" * 68)
    ans = input("\nConfirm start? (Enter / q to quit): ").strip().lower()
    if ans == "q":
        return

    records = []
    global_pulse = 0
    inst = connect(args.address)
    t_zero = time.time()
    try:
        init_smu(inst, args.icc, args.nplc)
        current, resistance, conductance = read_resistance(inst, args.v_read, args.n_read_avg, args.read_settle)
        add_record(records, 0, "baseline", 0, global_pulse, time.time() - t_zero, 0.0, current, resistance, conductance)
        print(f"\n[Baseline] R={fmt_resistance(resistance)}  G={conductance*1e6:.3f} uS")

        for cycle in range(1, args.cycles + 1):
            global_pulse = run_phase(
                inst, args, records, cycle, "potentiation", args.v_pot, args.n_pot, global_pulse, t_zero
            )
            global_pulse = run_phase(
                inst, args, records, cycle, "depression", args.v_dep, args.n_dep, global_pulse, t_zero
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
    finite_g = np.array([r["conductance_S"] for r in records if np.isfinite(r["conductance_S"])], dtype=float)
    if len(finite_r):
        print(f"[Summary] R range: {fmt_resistance(float(np.min(finite_r)))} to {fmt_resistance(float(np.max(finite_r)))}")
        if np.min(finite_r) > 0:
            print(f"[Summary] Rmax/Rmin: {float(np.max(finite_r) / np.min(finite_r)):.3f}x")
    if len(finite_g) > 1:
        print(f"[Summary] G span: {(float(np.max(finite_g) - np.min(finite_g)) * 1e6):.3f} uS")

    if not args.no_plot:
        title = f"{args.dev} pulse modulation: POT {args.v_pot:+.2f} V, DEP {args.v_dep:+.2f} V"
        plot_records(records, img_path, title)


if __name__ == "__main__":
    main()

