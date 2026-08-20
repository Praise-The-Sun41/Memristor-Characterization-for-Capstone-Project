"""
Replot Trial 4 IV sweep CSV files with corrected resistance in Icc regions.

Raw CSV files are not modified. Existing PNG files with matching names are
overwritten.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "IV_sweep" / "Trial 4 Results Ti AlOx TiOx AlOx Ni"
CYCLE_PALETTE = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]


def load_sweep_csv(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"empty CSV: {path}")

    def arr(name):
        return np.array([float(r[name]) for r in rows], dtype=float)

    def iarr(name):
        return np.array([int(float(r[name])) for r in rows], dtype=int)

    data = {
        "index": iarr("index"),
        "cycle": iarr("cycle"),
        "index_in_cycle": iarr("index_in_cycle"),
        "time_s": arr("time_s"),
        "voltage_V": arr("voltage_V"),
        "current_A": arr("current_A"),
        "icc_A": arr("icc_A"),
    }
    data["n_cycles"] = int(np.max(data["cycle"])) + 1
    data["cycle_len"] = int(np.max(data["index_in_cycle"])) + 1
    return data


def corrected_resistance(voltages, currents, icc_used, trigger_ratio=0.985):
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


def icc_by_polarity(voltages, icc_used):
    pos = icc_used[voltages >= 0]
    neg = icc_used[voltages < 0]
    pos_icc = float(np.nanmedian(pos)) if len(pos) else float(np.nanmedian(icc_used))
    neg_icc = float(np.nanmedian(neg)) if len(neg) else pos_icc
    return pos_icc, neg_icc


def fmt_icc(icc):
    return f"{icc * 1e6:.0f} uA" if icc < 1e-3 else f"{icc * 1e3:.1f} mA"


def style_axes(*axes):
    for ax in axes:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.grid(True, color="#2A2D37", lw=0.5, which="both", zorder=0)


def add_icc_lines(ax, voltages, icc_used):
    pos_icc, neg_icc = icc_by_polarity(voltages, icc_used)
    ax.axhline(pos_icc * 1e6, color="#FF5722", lw=0.9, linestyle="--",
               alpha=0.75, label=f"+Icc = {fmt_icc(pos_icc)}")
    ax.axhline(-neg_icc * 1e6, color="#FF9800", lw=0.9, linestyle="--",
               alpha=0.75, label=f"-Icc = {fmt_icc(neg_icc)}")


def smooth_median(values, win):
    smoothed = np.full(len(values), np.nan)
    half = win // 2
    for k in range(len(values)):
        chunk = values[max(0, k - half):min(len(values), k + half + 1)]
        valid = chunk[np.isfinite(chunk)]
        if len(valid):
            smoothed[k] = np.nanmedian(valid)
    return smoothed


def plot_summary(data, resistance, img_path: Path, title_suffix: str):
    v = data["voltage_V"]
    i_uA = data["current_A"] * 1e6
    icc = data["icc_A"]
    n_cycles = data["n_cycles"]
    cycle_len = data["cycle_len"]
    r_k = resistance / 1e3
    idx = np.arange(len(v))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")
    style_axes(ax1, ax2)

    for cyc in range(n_cycles):
        mask = data["cycle"] == cyc
        col = CYCLE_PALETTE[cyc % len(CYCLE_PALETTE)]
        ax1.plot(v[mask], i_uA[mask], color=col, lw=1.5, alpha=0.85, label=f"Cycle {cyc + 1}")
        ax2.scatter(idx[mask], r_k[mask], color=col, s=5, alpha=0.7)
        sm = smooth_median(r_k[mask], max(5, int(cycle_len * 0.08)))
        ax2.plot(idx[mask], sm, color=col, lw=1.2, alpha=0.9)

    add_icc_lines(ax1, v, icc)
    ax1.axhline(0, color="#555", lw=0.6)
    ax1.axvline(0, color="#555", lw=0.6)
    ax1.set_xlabel("Voltage (V)", color="#CCCCCC")
    ax1.set_ylabel("Current (uA)", color="#CCCCCC")
    ax1.set_title("I-V Characteristics", color="#EEEEEE", fontsize=11)
    ax1.legend(facecolor="#1A1D27", edgecolor="#444", labelcolor="#CCCCCC", fontsize=7.5)

    for cyc in range(1, n_cycles):
        ax2.axvline(cyc * cycle_len, color="#555", lw=0.7, linestyle=":", alpha=0.6)
    ax2.set_yscale("log")
    ax2.set_xlabel("Sample Index", color="#CCCCCC")
    ax2.set_ylabel("Resistance with Icc hold (kOhm)", color="#CCCCCC")
    ax2.set_title("Resistance vs. Sample Index", color="#EEEEEE", fontsize=11)

    fig.suptitle(f"IV Sweep Replot with Icc Hold   {title_suffix}", color="#DDDDDD", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cycle(data, resistance, cycle_number: int, img_path: Path, title_suffix: str):
    mask = data["cycle"] == cycle_number
    v = data["voltage_V"][mask]
    i_uA = data["current_A"][mask] * 1e6
    icc = data["icc_A"][mask]
    r_k = resistance[mask] / 1e3
    x = np.arange(len(v))
    col = CYCLE_PALETTE[cycle_number % len(CYCLE_PALETTE)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0F1117")
    style_axes(ax1, ax2)

    cycle_len = len(v)
    segs = [
        (0, cycle_len // 4, "#69F0AE", "0 to Vmax"),
        (cycle_len // 4, cycle_len // 2, "#FFD54F", "Vmax to 0"),
        (cycle_len // 2, 3 * cycle_len // 4, "#80DEEA", "0 to Vmin"),
        (3 * cycle_len // 4, cycle_len, "#EF9A9A", "Vmin to 0"),
    ]
    for lo, hi, seg_col, label in segs:
        ax1.plot(v[lo:hi], i_uA[lo:hi], color=seg_col, lw=2.0, alpha=0.9, label=label)

    add_icc_lines(ax1, v, icc)
    ax1.axhline(0, color="#555", lw=0.6)
    ax1.axvline(0, color="#555", lw=0.6)
    ax1.set_xlabel("Voltage (V)", color="#CCCCCC")
    ax1.set_ylabel("Current (uA)", color="#CCCCCC")
    ax1.set_title(f"I-V | Cycle {cycle_number + 1}", color="#EEEEEE", fontsize=11)
    ax1.legend(facecolor="#1A1D27", edgecolor="#444", labelcolor="#CCCCCC", fontsize=7.5)

    finite = np.isfinite(r_k)
    ax2.scatter(x[finite], r_k[finite], color=col, s=7, alpha=0.75)
    ax2.plot(x, smooth_median(r_k, max(5, int(cycle_len * 0.08))),
             color=col, lw=1.6, alpha=0.9)
    for sep in [cycle_len // 4, cycle_len // 2, 3 * cycle_len // 4]:
        ax2.axvline(sep, color="#555", lw=0.7, linestyle=":", alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_xlabel("Sample Index (within cycle)", color="#CCCCCC")
    ax2.set_ylabel("Resistance with Icc hold (kOhm)", color="#CCCCCC")
    ax2.set_title(f"Resistance | Cycle {cycle_number + 1}", color="#EEEEEE", fontsize=11)

    fig.suptitle(f"Cycle {cycle_number + 1} Replot with Icc Hold   {title_suffix}",
                 color="#DDDDDD", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def replot_one(csv_path: Path):
    data = load_sweep_csv(csv_path)
    resistance = corrected_resistance(data["voltage_V"], data["current_A"], data["icc_A"])
    stem = csv_path.stem
    ts = stem.replace("iv_sweep_", "")
    title_suffix = csv_path.parent.name

    summary_png = csv_path.with_suffix(".png")
    plot_summary(data, resistance, summary_png, title_suffix)

    for cyc in range(data["n_cycles"]):
        cycle_png = csv_path.parent / f"iv_cycle{cyc + 1}_{ts}.png"
        plot_cycle(data, resistance, cyc, cycle_png, title_suffix)

    return 1 + data["n_cycles"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="Trial 4 root folder")
    args = parser.parse_args()

    root = Path(args.root)
    sweep_csvs = sorted(root.rglob("iv_sweep_*.csv"))
    if not sweep_csvs:
        raise SystemExit(f"No iv_sweep_*.csv found under {root}")

    image_count = 0
    for csv_path in sweep_csvs:
        print(f"[Replot] {csv_path}")
        image_count += replot_one(csv_path)

    print(f"[Done] Replotted {len(sweep_csvs)} runs, wrote {image_count} PNG files.")


if __name__ == "__main__":
    main()
