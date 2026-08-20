"""
./IV_sweep/V-5.0_5.0_cyc5_pts50_NPLC0.5_20260519_145944
"""
"""
plot_iv_bulk.py
───────────────
Bulk-reads all iv_cycle*.csv files in a folder and plots |I| vs V
on a semi-log Y axis, one curve per cycle, styled after literature figures.

Usage
-----
  python plot_iv_bulk.py                      # reads CSVs from current dir
  python plot_iv_bulk.py --folder ./data      # specify folder
  python plot_iv_bulk.py --folder ./data --out iv_summary.png
  python plot_iv_bulk.py --folder ./data --yscale linear

Output
------
  One PNG/PDF with:
    (a) Main panel  – |I| vs V, log Y, each cycle a separate coloured curve
    (b) Inset panel – OFF/ON ratio vs cycle index at a fixed read voltage
"""

import argparse
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Publication-style rcParams ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "DejaVu Sans"],
    "font.size":          11,
    "axes.linewidth":     1.2,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.minor.width":  0.8,
    "ytick.minor.width":  0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "legend.frameon":     True,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csvs(folder: str) -> dict[str, pd.DataFrame]:
    """
    Load all iv_cycle*.csv files from folder.
    Returns an ordered dict  {filename_stem: DataFrame}.
    Sorted by the numeric cycle index embedded in the filename.
    """
    pattern = os.path.join(folder, "iv_cycle*.csv")
    paths   = sorted(glob.glob(pattern),
                     key=lambda p: _extract_number(os.path.basename(p)))
    if not paths:
        raise FileNotFoundError(f"No iv_cycle*.csv files found in: {folder}")

    print(f"Found {len(paths)} file(s):")
    data = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        df   = pd.read_csv(p)
        # Normalise column names to lower-case stripped strings
        df.columns = [c.strip().lower() for c in df.columns]
        data[stem] = df
        print(f"  {os.path.basename(p)}  –  {len(df)} rows, "
              f"cycle id(s): {sorted(df['cycle'].unique())}")
    return data


def _extract_number(name: str) -> int:
    """Pull the first integer out of a filename for natural sort."""
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 0


def _abs_current(df: pd.DataFrame) -> np.ndarray:
    """Return |I| handling zero exactly (avoid log(0))."""
    col = "current_a" if "current_a" in df.columns else "current_ua"
    scale = 1.0 if col == "current_a" else 1e-6
    return np.abs(df[col].values * scale)


def build_colormap(n: int, cmap_name: str = "plasma") -> list:
    """n distinct colours from a matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_iv(data: dict[str, pd.DataFrame],
            read_v: float = 0.5,
            out_path: str = "iv_bulk_plot.png",
            yscale: str = "log") -> None:

    stems  = list(data.keys())
    n      = len(stems)
    colors = build_colormap(n)

    # ── figure layout: main axes + inset ────────────────────────────────────
    fig = plt.figure(figsize=(9.0, 7.0))
    ax  = fig.add_axes([0.11, 0.11, 0.84, 0.82])   # main (larger)

    # Inset: R_HRS / R_LRS scatter — placed bottom-right, clear of main curves
    # Original size was 0.38 × 0.30; shrink 30% → 0.266 × 0.210
    ax_ins = fig.add_axes([0.625, 0.145, 0.266, 0.210])  # [left,bot,w,h]

    read_v_hrs  = []   # R_HRS at read_v per file  (max resistance found near read_v)
    read_v_lrs  = []   # R_LRS at read_v per file  (min resistance found near read_v)
    cycle_labels = []

    for idx, (stem, df) in enumerate(data.items()):
        v   = df["voltage_v"].values
        iabs = _abs_current(df)

        # Replace exact zeros with a small floor so log is well-defined
        floor = 1e-12
        iabs  = np.where(iabs < floor, floor, iabs)

        color = colors[idx]

        # Label: try to extract a short cycle number from the stem
        m = re.search(r"cycle(\d+)", stem, re.IGNORECASE)
        label = f"cycle {m.group(1)}" if m else stem[-12:]

        if yscale == "log":
            ax.semilogy(v, iabs, color=color, linewidth=1.4,
                        label=label, alpha=0.9)
        else:
            ax.plot(v, iabs, color=color, linewidth=1.4,
                    label=label, alpha=0.9)

        # Collect HRS and LRS current at read_v for OFF/ON inset
        # HRS  = minimum |I| found anywhere near +read_v on the return sweep
        # LRS  = maximum |I| found anywhere near +read_v on the forward sweep
        # Simple heuristic: split sweep at voltage maximum
        _collect_hrs_lrs(v, iabs, read_v, read_v_hrs, read_v_lrs)
        cycle_labels.append(label)

    # ── main axes cosmetics ──────────────────────────────────────────────────
    ax.set_xlabel("Voltage (V)", fontsize=12)
    ax.set_ylabel("|Current| (A)", fontsize=12)
    ax.set_xlim(v.min() * 1.05, v.max() * 1.05)
    if yscale == "log":
        ax.yaxis.set_major_formatter(
            ticker.LogFormatterMathtext(base=10, labelOnlyBase=True))
    else:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # Colour-bar style legend if many cycles, otherwise normal legend
    if n <= 8:
        leg = ax.legend(loc="upper left", ncol=1, handlelength=1.6,
                        borderpad=0.6, labelspacing=0.3)
    else:
        # Show only first / last few to avoid clutter
        handles, labels = ax.get_legend_handles_labels()
        keep = list(range(min(3, n))) + list(range(max(3, n - 2), n))
        keep = sorted(set(keep))
        leg = ax.legend([handles[i] for i in keep],
                        [labels[i]  for i in keep],
                        loc="upper left", ncol=1,
                        title=f"{n} cycles (subset shown)",
                        handlelength=1.6, borderpad=0.6, labelspacing=0.3)

    ax.set_title(f"I-V characteristics  (|I| vs V, {yscale} Y)",
                 fontsize=12, pad=6)

    # ── inset axes: R_HRS and R_LRS vs cycle index ──────────────────────────
    x_idx = np.arange(n)

    # _collect_hrs_lrs now returns resistance values directly
    r_hrs = np.array(read_v_hrs, dtype=float)
    r_lrs = np.array(read_v_lrs, dtype=float)

    valid = ~(np.isnan(r_hrs) | np.isnan(r_lrs))

    if valid.any():
        xi = x_idx[valid]

        r_all = np.concatenate([r_hrs[valid], r_lrs[valid]])
        r_data_min = r_all.min()
        r_data_max = r_all.max()

        # ── auto-scale: pad one decade below min and above max,
        #    but always show at least 2 decades so ratio lines are readable
        log_lo = np.floor(np.log10(r_data_min)) - 0.5
        log_hi = np.ceil (np.log10(r_data_max)) + 0.5
        if (log_hi - log_lo) < 2.0:          # enforce minimum 2-decade span
            mid = (log_lo + log_hi) / 2
            log_lo, log_hi = mid - 1.0, mid + 1.0
        r_min_ax = 10 ** log_lo
        r_max_ax = 10 ** log_hi

        # ── which ratio reference lines to draw?
        #    pick levels that fall within or just above the plot range
        actual_ratio = r_data_max / max(r_data_min, 1e-3)
        all_levels   = [2, 5, 10, 100, 1000, 1e4]
        show_levels  = [lv for lv in all_levels
                        if lv <= actual_ratio * 20]   # show up to 20× the real ratio
        if not show_levels:
            show_levels = [10]

        for ratio_level in show_levels:
            y_band = r_lrs[valid] * ratio_level
            # only draw where y_band is within plot range
            in_range = (y_band >= r_min_ax * 0.5) & (y_band <= r_max_ax * 2)
            if not in_range.any():
                continue
            ax_ins.semilogy(xi, y_band, linestyle="--", color="0.60",
                            linewidth=0.7, alpha=0.45, zorder=1)
            # label at the rightmost in-range point
            last = np.where(in_range)[0][-1]
            label_str = (f"×{int(ratio_level)}" if ratio_level >= 2
                         else f"×{ratio_level:.1f}")
            ax_ins.text(xi[last] + 0.15, y_band[last],
                        label_str, fontsize=5.5,
                        color="0.50", va="center", ha="left", zorder=1)

        # ── scatter: HRS (circles) and LRS (triangles) ───────────────────────
        ax_ins.semilogy(xi, r_hrs[valid], "o", color="#185FA5",
                        markersize=5, markeredgecolor="white",
                        markeredgewidth=0.4, zorder=4,
                        label=f"$R_{{HRS}}$")
        ax_ins.semilogy(xi, r_lrs[valid], "^", color="#0F6E56",
                        markersize=5, markeredgecolor="white",
                        markeredgewidth=0.4, zorder=4,
                        label=f"$R_{{LRS}}$")

        # thin connecting lines between paired points
        for xi_, rh, rl in zip(xi, r_hrs[valid], r_lrs[valid]):
            ax_ins.semilogy([xi_, xi_], [rl, rh],
                            color="0.78", linewidth=0.5, zorder=2)

        ax_ins.set_xlabel("Cycle index", fontsize=7)
        ax_ins.set_ylabel("R  (Ω)", fontsize=7)
        ax_ins.tick_params(labelsize=6, direction="in",
                           top=True, right=True, length=2.5, pad=2)
        ax_ins.set_title(f"$R_{{HRS}}$ & $R_{{LRS}}$  @  {read_v} V",
                         fontsize=7, pad=2.5)
        ax_ins.set_xlim(-0.6, n - 0.4)
        ax_ins.set_ylim(r_min_ax, r_max_ax)
        ax_ins.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        ax_ins.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax_ins.legend(fontsize=6, framealpha=0.85, loc="best",
                      handletextpad=0.3, borderpad=0.4,
                      handlelength=1.0, markerscale=0.9)
        for spine in ax_ins.spines.values():
            spine.set_linewidth(0.7)
        ax_ins.set_facecolor((1, 1, 1, 0.88))

    plt.savefig(out_path)
    print(f"\nSaved → {out_path}")
    plt.close(fig)


def _collect_hrs_lrs(v_arr, i_arr, target_v, hrs_list, lrs_list):
    """
    Find HRS and LRS resistance at target_v by looking at ALL points
    within a voltage window around target_v, then taking the max and min
    resistance values found there.

    HRS = highest resistance  (lowest current)
    LRS = lowest  resistance  (highest current)

    This is direction-agnostic: works for both bipolar sweeps and
    single-direction sweeps, and doesn't assume which branch is LRS/HRS.
    """
    # Use a ±10% window around target_v to collect candidate points
    tol    = max(abs(target_v) * 0.15, 0.05)   # at least 50 mV window
    mask   = (np.abs(np.abs(v_arr) - abs(target_v)) <= tol) & (np.abs(v_arr) > 0)

    if mask.sum() < 1:
        hrs_list.append(np.nan)
        lrs_list.append(np.nan)
        return

    # Resistance = |V| / |I| at each matching point
    v_sel  = np.abs(v_arr[mask])
    i_sel  = np.abs(i_arr[mask])

    # Avoid division by zero
    good   = i_sel > 1e-13
    if good.sum() < 1:
        hrs_list.append(np.nan)
        lrs_list.append(np.nan)
        return

    r_vals = v_sel[good] / i_sel[good]

    hrs_list.append(float(r_vals.max()))   # HRS = highest R  (lowest I)
    lrs_list.append(float(r_vals.min()))   # LRS = lowest  R  (highest I)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default="./IV_sweep/Trial 4 Results Ti AlOx TiOx AlOx Ni/Dev 2-3 valid+/V-5.0_5.0_cyc2_pts100_SET50uA_RST5mA_NPLC0.1_20260814_165940",
                        help="Folder containing iv_cycle*.csv files (default: .)")
    parser.add_argument("--out",    default="iv_bulk_plot.png",
                        help="Output filename (default: iv_bulk_plot.png)")
    parser.add_argument("--read_v", default=0.5, type=float,
                        help="Read voltage for inset panel (default: 0.5 V)")
    parser.add_argument("--yscale", choices=("log", "linear"), default="log",
                        help="Y-axis scale for main |I|-V plot: log or linear (default: log)")
    args = parser.parse_args()

    data = load_csvs(args.folder)
    plot_iv(data, read_v=args.read_v, out_path=args.out, yscale=args.yscale)


if __name__ == "__main__":
    main()
