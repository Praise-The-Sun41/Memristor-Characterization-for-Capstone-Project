"""
Offline convolution simulation from measured I-V data.

This script reuses existing I-V CSV files to build an empirical conductance
state library, simulates writing a target convolution kernel into a differential
memristor array, and compares:

  1. experimental-data result: kernel weights quantized to measured states
  2. theoretical result: ideal continuous weights

Examples
--------
  python iv_convolution_sim.py
  python iv_convolution_sim.py --list-devices
  python iv_convolution_sim.py --device 1 --pulse-device 2 --read-v 0.1
  python iv_convolution_sim.py --input "[[1,2,3],[4,5,6],[7,8,9]]" \
      --kernel "[[1,0],[-1,0]]"
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    PLOT_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MEASURE_ROOT = PROJECT_ROOT / "memristor_measure"
DEFAULT_IV_ROOT = MEASURE_ROOT / "IV_sweep"
DEFAULT_PULSE_ROOT = MEASURE_ROOT / "Pulse_LTP_LTD"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "Convolution_Sim"


DEFAULT_KERNEL = np.array(
    [
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ],
    dtype=float,
)

DEFAULT_RANDOM_ROWS = 8
DEFAULT_RANDOM_COLS = 8
DEFAULT_RANDOM_LOW = 0
DEFAULT_RANDOM_HIGH = 9
DEFAULT_RANDOM_SEED = 7


@dataclass(frozen=True)
class ConductanceLibrary:
    states_s: np.ndarray
    source_files: int
    source_points: int
    read_v: float
    tolerance_v: float

    @property
    def g_min(self) -> float:
        return float(np.min(self.states_s))

    @property
    def g_max(self) -> float:
        return float(np.max(self.states_s))

    @property
    def dynamic_range(self) -> float:
        if self.g_min <= 0:
            return math.inf
        return self.g_max / self.g_min


@dataclass(frozen=True)
class PulseLibrary:
    states_s: np.ndarray
    ltp_delta_s: np.ndarray
    ltd_delta_s: np.ndarray
    source_files: int
    source_points: int

    @property
    def available(self) -> bool:
        return self.states_s.size > 1 and self.ltp_delta_s.size > 0 and self.ltd_delta_s.size > 0


def parse_matrix(text: str | None, default: np.ndarray) -> np.ndarray:
    if not text:
        return default.copy()
    value = ast.literal_eval(text)
    arr = np.array(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix arguments must be 2-D lists")
    return arr


def random_input_matrix(
    rows: int,
    cols: int,
    low: int,
    high: int,
    seed: int | None,
) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise ValueError("random input dimensions must be positive")
    if high < low:
        raise ValueError("--random-high must be greater than or equal to --random-low")
    rng = np.random.default_rng(seed)
    return rng.integers(low, high + 1, size=(rows, cols)).astype(float)


def path_matches_device(path: Path, device: str | None) -> bool:
    if not device:
        return True
    return device.lower() in str(path).lower()


def iv_device_candidates(iv_root: Path) -> list[Path]:
    if not iv_root.exists():
        return []
    candidates = []
    for path in sorted(p for p in iv_root.rglob("*") if p.is_dir()):
        name = path.name.lower()
        if "device" in name or "dev" in name or "unit" in name:
            if any(path.rglob("*.csv")):
                candidates.append(path)
    return candidates


def pulse_device_candidates(pulse_root: Path) -> list[Path]:
    if not pulse_root.exists():
        return []
    return sorted(p for p in pulse_root.iterdir() if p.is_dir() and any(p.glob("*.csv")))


def resolve_numbered_choice(value: str | None, candidates: list[Path], root: Path) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        idx = int(value)
        if idx < 1 or idx > len(candidates):
            raise ValueError(f"choice {idx} is out of range; run --list-devices")
        return str(candidates[idx - 1].relative_to(root))
    return value


def iter_csv_files(root: Path, device: str | None = None) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".csv":
        if path_matches_device(root, device):
            yield root
        return
    for path in root.rglob("*.csv"):
        if path_matches_device(path, device):
            yield path


def find_column(columns: list[str], candidates: tuple[str, ...]) -> int | None:
    normalized = [c.strip().lower() for c in columns]
    for name in candidates:
        if name in normalized:
            return normalized.index(name)
    return None


def load_iv_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return voltage and current arrays from either project or LabVIEW CSVs."""
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.reader(f))

    for header_idx, row in enumerate(rows):
        lower = [c.strip().lower() for c in row]
        v_idx = find_column(row, ("voltage_v", "voltage", "v", "value"))
        i_idx = find_column(row, ("current_a", "current", "i", "reading"))
        iu_idx = find_column(row, ("current_ua", "current_microa"))

        if v_idx is None or (i_idx is None and iu_idx is None):
            continue

        # Keithley/LabVIEW export: Reading is current and Value is voltage.
        if "reading" in lower and "value" in lower:
            i_idx = lower.index("reading")
            v_idx = lower.index("value")

        voltage = []
        current = []
        for data_row in rows[header_idx + 1 :]:
            if max(v_idx, i_idx if i_idx is not None else iu_idx) >= len(data_row):
                continue
            try:
                v = float(data_row[v_idx])
                if i_idx is not None:
                    i = float(data_row[i_idx])
                else:
                    i = float(data_row[iu_idx]) * 1e-6
            except (TypeError, ValueError):
                continue
            voltage.append(v)
            current.append(i)

        if voltage and current:
            return np.array(voltage, dtype=float), np.array(current, dtype=float)

    raise ValueError(f"no voltage/current columns found in {path}")


def build_conductance_library(
    iv_root: Path,
    read_v: float,
    tolerance_v: float,
    min_abs_v: float,
    max_files: int | None,
    device: str | None,
) -> ConductanceLibrary:
    states = []
    source_files = 0
    source_points = 0

    for path in iter_csv_files(iv_root, device=device):
        if max_files is not None and source_files >= max_files:
            break
        try:
            voltage, current = load_iv_csv(path)
        except Exception:
            continue

        abs_v = np.abs(voltage)
        if read_v > 0:
            mask = np.abs(abs_v - abs(read_v)) <= tolerance_v
        else:
            mask = abs_v >= min_abs_v

        if not np.any(mask):
            nearest = np.argsort(np.abs(abs_v - abs(read_v)))[: min(4, len(abs_v))]
            mask = np.zeros_like(abs_v, dtype=bool)
            mask[nearest] = True

        valid = mask & (abs_v >= min_abs_v) & np.isfinite(current)
        conductance = np.abs(current[valid]) / abs_v[valid]
        conductance = conductance[np.isfinite(conductance) & (conductance > 0)]
        if conductance.size == 0:
            continue

        states.extend(conductance.tolist())
        source_files += 1
        source_points += int(conductance.size)

    if not states:
        raise RuntimeError(f"no usable conductance states found under {iv_root}")

    states_arr = np.array(states, dtype=float)
    lo, hi = np.percentile(states_arr, [1, 99])
    states_arr = states_arr[(states_arr >= lo) & (states_arr <= hi)]
    states_arr = np.unique(np.round(states_arr, decimals=12))
    states_arr.sort()

    if states_arr.size < 2:
        raise RuntimeError("not enough distinct conductance states for simulation")

    return ConductanceLibrary(
        states_s=states_arr,
        source_files=source_files,
        source_points=source_points,
        read_v=read_v,
        tolerance_v=tolerance_v,
    )


def load_pulse_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"no header found in {path}")
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        phase_col = field_map.get("phase")
        g_col = field_map.get("g_s")
        if phase_col is None or g_col is None:
            raise ValueError(f"no phase/G_S columns found in {path}")

        states = []
        ltp = []
        ltd = []
        prev_by_phase: dict[str, float] = {}
        for row in reader:
            phase = row.get(phase_col, "").strip().upper()
            try:
                g = float(row[g_col])
            except (TypeError, ValueError):
                continue
            if phase not in ("LTP", "LTD"):
                continue
            if g > 0:
                states.append(g)
            prev = prev_by_phase.get(phase)
            if prev is not None:
                delta = g - prev
                if phase == "LTP" and delta > 0:
                    ltp.append(delta)
                elif phase == "LTD" and delta < 0:
                    ltd.append(delta)
            prev_by_phase[phase] = g

    return np.array(states, dtype=float), np.array(ltp, dtype=float), np.array(ltd, dtype=float)


def build_pulse_library(
    pulse_root: Path,
    device: str | None,
    max_files: int | None,
) -> PulseLibrary:
    states_all = []
    ltp_all = []
    ltd_all = []
    source_files = 0
    source_points = 0

    for path in iter_csv_files(pulse_root, device=device):
        if max_files is not None and source_files >= max_files:
            break
        if not path.name.lower().startswith("ltp_ltd_"):
            continue
        try:
            states, ltp, ltd = load_pulse_csv(path)
        except Exception:
            continue
        if states.size == 0 and ltp.size == 0 and ltd.size == 0:
            continue
        states_all.extend(states.tolist())
        ltp_all.extend(ltp.tolist())
        ltd_all.extend(ltd.tolist())
        source_files += 1
        source_points += int(states.size)

    states_arr = np.array(states_all, dtype=float)
    if states_arr.size:
        lo, hi = np.percentile(states_arr, [1, 99])
        states_arr = states_arr[(states_arr >= lo) & (states_arr <= hi)]
        states_arr = np.unique(np.round(states_arr, decimals=15))
        states_arr.sort()

    return PulseLibrary(
        states_s=states_arr,
        ltp_delta_s=np.array(ltp_all, dtype=float),
        ltd_delta_s=np.array(ltd_all, dtype=float),
        source_files=source_files,
        source_points=source_points,
    )


def make_closed_loop_write_library(
    pulse_lib: PulseLibrary,
    read_v: float,
    tolerance_v: float,
    max_write_pulses: int,
) -> ConductanceLibrary:
    g_low = float(np.percentile(pulse_lib.states_s, 1))
    g_high_data = float(np.percentile(pulse_lib.states_s, 95))
    ltp_step = float(np.median(pulse_lib.ltp_delta_s))
    g_high_budget = g_low + max_write_pulses * max(ltp_step, 0.0)
    g_high = min(g_high_data, g_high_budget)
    if g_high <= g_low:
        g_high = float(np.max(pulse_lib.states_s))
    states = np.array([g_low, g_high], dtype=float)
    return ConductanceLibrary(
        states_s=states,
        source_files=pulse_lib.source_files,
        source_points=pulse_lib.source_points,
        read_v=read_v,
        tolerance_v=tolerance_v,
    )


def nearest_state(states: np.ndarray, target: np.ndarray) -> np.ndarray:
    flat = target.ravel()
    idx = np.searchsorted(states, flat)
    idx = np.clip(idx, 1, len(states) - 1)
    left = states[idx - 1]
    right = states[idx]
    choose_right = np.abs(right - flat) < np.abs(flat - left)
    out = np.where(choose_right, right, left)
    return out.reshape(target.shape)


def simulate_closed_loop_write(
    target_s: np.ndarray,
    lib: ConductanceLibrary,
    pulse_lib: PulseLibrary,
    tolerance_s: float,
    max_pulses: int,
) -> tuple[np.ndarray, list[dict[str, float | int | str | bool]]]:
    if not pulse_lib.available:
        return nearest_state(lib.states_s, target_s), []

    ltp_step = float(np.median(pulse_lib.ltp_delta_s))
    ltd_step = float(np.median(pulse_lib.ltd_delta_s))
    if ltp_step <= 0 or ltd_step >= 0:
        return nearest_state(lib.states_s, target_s), []

    final = np.zeros_like(target_s, dtype=float)
    records: list[dict[str, float | int | str | bool]] = []

    for index, target in np.ndenumerate(target_s):
        g = lib.g_min
        pulses = 0
        last_phase = "INIT"

        for _ in range(max_pulses):
            err = target - g
            if abs(err) <= tolerance_s:
                break

            if err > 0:
                g += ltp_step
                last_phase = "LTP"
            else:
                g += ltd_step
                last_phase = "LTD"

            g = min(max(g, lib.g_min), lib.g_max)
            pulses += 1

        final[index] = g
        records.append(
            {
                "row": int(index[0]),
                "col": int(index[1]),
                "target_s": float(target),
                "final_s": float(g),
                "error_s": float(g - target),
                "pulses": int(pulses),
                "last_phase": last_phase,
                "hit_tolerance": bool(abs(g - target) <= tolerance_s),
            }
        )

    return final, records


def encode_differential_kernel(
    kernel: np.ndarray,
    lib: ConductanceLibrary,
    write_method: str,
    pulse_lib: PulseLibrary | None,
    write_tolerance_s: float,
    max_write_pulses: int,
) -> dict[str, np.ndarray | float]:
    scale = float(np.max(np.abs(kernel)))
    if scale == 0:
        raise ValueError("kernel cannot be all zeros")

    normalized = kernel / scale
    span = lib.g_max - lib.g_min
    pos_target = lib.g_min + np.clip(normalized, 0, 1) * span
    neg_target = lib.g_min + np.clip(-normalized, 0, 1) * span

    write_records = []
    if write_method == "closed-loop" and pulse_lib is not None and pulse_lib.available:
        pos_exp, pos_records = simulate_closed_loop_write(
            pos_target, lib, pulse_lib, write_tolerance_s, max_write_pulses
        )
        neg_exp, neg_records = simulate_closed_loop_write(
            neg_target, lib, pulse_lib, write_tolerance_s, max_write_pulses
        )
        for item in pos_records:
            item["array"] = "G_pos"
        for item in neg_records:
            item["array"] = "G_neg"
        write_records = pos_records + neg_records
    else:
        pos_exp = nearest_state(lib.states_s, pos_target)
        neg_exp = nearest_state(lib.states_s, neg_target)

    effective_kernel = (pos_exp - neg_exp) / span * scale
    theoretical_kernel = (pos_target - neg_target) / span * scale

    return {
        "scale": scale,
        "pos_target_s": pos_target,
        "neg_target_s": neg_target,
        "pos_exp_s": pos_exp,
        "neg_exp_s": neg_exp,
        "effective_kernel": effective_kernel,
        "theoretical_kernel": theoretical_kernel,
        "write_records": write_records,
    }


def convolve2d(input_matrix: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    kh, kw = kernel.shape
    if kh > input_matrix.shape[0] or kw > input_matrix.shape[1]:
        raise ValueError("kernel must not be larger than input for valid convolution")

    if mode == "same":
        pad_h = kh // 2
        pad_w = kw // 2
        work = np.pad(input_matrix, ((pad_h, kh - pad_h - 1), (pad_w, kw - pad_w - 1)))
    elif mode == "valid":
        work = input_matrix
    else:
        raise ValueError("mode must be 'valid' or 'same'")

    out_h = work.shape[0] - kh + 1
    out_w = work.shape[1] - kw + 1
    out = np.zeros((out_h, out_w), dtype=float)
    flipped = np.flipud(np.fliplr(kernel))
    for r in range(out_h):
        for c in range(out_w):
            out[r, c] = float(np.sum(work[r : r + kh, c : c + kw] * flipped))
    return out


def error_metrics(experimental: np.ndarray, theoretical: np.ndarray) -> dict[str, float]:
    err = experimental - theoretical
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.max(np.abs(theoretical)))
    max_abs_pct = float(np.max(np.abs(err)) / denom * 100) if denom > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "max_abs_pct": max_abs_pct}


def save_matrix_csv(path: Path, **matrices: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name, matrix in matrices.items():
            writer.writerow([name])
            writer.writerows(matrix)
            writer.writerow([])


def save_write_log(path: Path, records: list[dict[str, float | int | str | bool]]) -> None:
    if not records:
        return
    fields = [
        "array",
        "row",
        "col",
        "target_s",
        "final_s",
        "error_s",
        "pulses",
        "last_phase",
        "hit_tolerance",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def list_device_candidates(iv_root: Path, pulse_root: Path) -> None:
    print("[IV device candidates]")
    iv_candidates = iv_device_candidates(iv_root)
    if iv_candidates:
        for idx, path in enumerate(iv_candidates, start=1):
            print(f"  [{idx}] {path.relative_to(iv_root)}")
    else:
        print("  (none)")

    print("\n[Pulse device/config candidates]")
    pulse_candidates = pulse_device_candidates(pulse_root)
    if pulse_candidates:
        for idx, path in enumerate(pulse_candidates, start=1):
            print(f"  [{idx}] {path.relative_to(pulse_root)}")
    else:
        print("  (none)")


def plot_results(
    path: Path,
    input_matrix: np.ndarray,
    target_kernel: np.ndarray,
    effective_kernel: np.ndarray,
    experimental: np.ndarray,
    theoretical: np.ndarray,
) -> None:
    if not PLOT_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        ("Input", input_matrix),
        ("Target kernel", target_kernel),
        ("Experimental kernel", effective_kernel),
        ("Theoretical output", theoretical),
        ("Experimental output", experimental),
        ("Error", experimental - theoretical),
    ]

    for ax, (title, data) in zip(axes.ravel(), panels):
        im = ax.imshow(data, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(c, r, f"{data[r, c]:.2g}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_state_library(path: Path, lib: ConductanceLibrary) -> None:
    if not PLOT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lib.states_s * 1e6, bins=min(80, max(10, lib.states_s.size // 4)))
    ax.set_xlabel("Conductance (uS)")
    ax.set_ylabel("Count")
    ax.set_title("Measured conductance states used for write simulation")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate memristor-array convolution with measured I-V data and "
            "closed-loop pulse-write behavior."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Typical workflow
----------------
  1. List available numbered IV devices and pulse-write datasets:
       python iv_convolution_sim.py --list-devices

  2. Run closed-loop pulse-write convolution with numbered choices:
       python iv_convolution_sim.py --device 1 --pulse-device 5

  3. Use a larger reproducible random input matrix:
       python iv_convolution_sim.py --device 1 --random-rows 10 --random-cols 10 --seed 12

  4. Override the random input and kernel manually:
       python iv_convolution_sim.py --input "[[1,2,3],[4,5,6],[7,8,9]]" --kernel "[[1,0],[-1,0]]"

Device selection
----------------
  --device accepts either a number from --list-devices or a path substring.
  --pulse-device accepts either a number from --list-devices or a path substring.

Default paths
-------------
  IV data root      : {DEFAULT_IV_ROOT}
  Pulse data root   : {DEFAULT_PULSE_ROOT}
  Simulation output : {DEFAULT_OUT_ROOT}
"""
    )
    parser.add_argument("--iv-root", type=Path, default=DEFAULT_IV_ROOT,
                        help="Root folder or one CSV file for measured I-V data.")
    parser.add_argument("--pulse-root", type=Path, default=DEFAULT_PULSE_ROOT,
                        help="Root folder for LTP/LTD pulse-write CSV data.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output folder. Default creates Convolution_Sim/conv_sim_TIMESTAMP.")
    parser.add_argument("--device", default=None,
                        help="IV device number from --list-devices, or a path substring.")
    parser.add_argument("--pulse-device", default=None,
                        help="Pulse dataset number from --list-devices, or a path substring.")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print numbered IV and pulse candidates, then exit.")
    parser.add_argument("--input", dest="input_text", default=None,
                        help="Manual 2-D input matrix, e.g. \"[[1,2],[3,4]]\". Overrides random input.")
    parser.add_argument("--random-rows", type=int, default=DEFAULT_RANDOM_ROWS,
                        help="Rows for default random input matrix.")
    parser.add_argument("--random-cols", type=int, default=DEFAULT_RANDOM_COLS,
                        help="Columns for default random input matrix.")
    parser.add_argument("--random-low", type=int, default=DEFAULT_RANDOM_LOW,
                        help="Inclusive lower bound for random input values.")
    parser.add_argument("--random-high", type=int, default=DEFAULT_RANDOM_HIGH,
                        help="Inclusive upper bound for random input values.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed for reproducible default input.")
    parser.add_argument("--kernel", dest="kernel_text", default=None,
                        help="Manual 2-D convolution kernel. Default is a Sobel-like edge kernel.")
    parser.add_argument("--mode", choices=("valid", "same"), default="valid",
                        help="Convolution mode.")
    parser.add_argument("--write-method", choices=("closed-loop", "nearest-state"), default="closed-loop",
                        help="closed-loop simulates pulse-by-pulse write/read/verify; nearest-state uses direct quantization.")
    parser.add_argument("--write-tolerance-us", type=float, default=0.005,
                        help="Closed-loop write tolerance in micro-Siemens.")
    parser.add_argument("--max-write-pulses", type=int, default=10,
                        help="Maximum pulse count per G_pos/G_neg cell.")
    parser.add_argument("--read-v", type=float, default=0.1,
                        help="Read voltage used to extract conductance from I-V data.")
    parser.add_argument("--tolerance-v", type=float, default=0.02,
                        help="Voltage window around --read-v when extracting I-V conductance.")
    parser.add_argument("--min-abs-v", type=float, default=1e-4,
                        help="Ignore points with |V| below this threshold.")
    parser.add_argument("--max-files", type=int, default=200,
                        help="Maximum CSV files to read from each data source.")
    args = parser.parse_args()

    if args.list_devices:
        list_device_candidates(args.iv_root, args.pulse_root)
        return

    iv_device_filter = resolve_numbered_choice(
        args.device, iv_device_candidates(args.iv_root), args.iv_root
    )
    pulse_device_filter = resolve_numbered_choice(
        args.pulse_device, pulse_device_candidates(args.pulse_root), args.pulse_root
    )

    default_input = random_input_matrix(
        args.random_rows,
        args.random_cols,
        args.random_low,
        args.random_high,
        args.seed,
    )
    input_matrix = parse_matrix(args.input_text, default_input)
    target_kernel = parse_matrix(args.kernel_text, DEFAULT_KERNEL)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / f"conv_sim_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    lib = build_conductance_library(
        args.iv_root,
        read_v=args.read_v,
        tolerance_v=args.tolerance_v,
        min_abs_v=args.min_abs_v,
        max_files=args.max_files,
        device=iv_device_filter,
    )
    pulse_lib = build_pulse_library(args.pulse_root, pulse_device_filter, args.max_files)
    write_method_used = args.write_method
    if args.write_method == "closed-loop" and not pulse_lib.available:
        write_method_used = "nearest-state"
    write_lib = lib
    if write_method_used == "closed-loop":
        write_lib = make_closed_loop_write_library(
            pulse_lib,
            read_v=args.read_v,
            tolerance_v=args.tolerance_v,
            max_write_pulses=args.max_write_pulses,
        )

    encoded = encode_differential_kernel(
        target_kernel,
        write_lib,
        write_method=write_method_used,
        pulse_lib=pulse_lib,
        write_tolerance_s=args.write_tolerance_us * 1e-6,
        max_write_pulses=args.max_write_pulses,
    )

    experimental_kernel = encoded["effective_kernel"]
    theoretical_kernel = encoded["theoretical_kernel"]
    experimental_output = convolve2d(input_matrix, experimental_kernel, args.mode)
    theoretical_output = convolve2d(input_matrix, theoretical_kernel, args.mode)
    metrics = error_metrics(experimental_output, theoretical_output)

    save_matrix_csv(
        out_dir / "convolution_results.csv",
        input=input_matrix,
        target_kernel=target_kernel,
        experimental_kernel=experimental_kernel,
        theoretical_output=theoretical_output,
        experimental_output=experimental_output,
        error=experimental_output - theoretical_output,
    )
    save_write_log(out_dir / "closed_loop_write_log.csv", encoded["write_records"])
    plot_results(
        out_dir / "convolution_results.png",
        input_matrix,
        target_kernel,
        experimental_kernel,
        experimental_output,
        theoretical_output,
    )
    plot_state_library(out_dir / "conductance_states.png", lib)

    summary = {
        "iv_root": str(args.iv_root),
        "pulse_root": str(args.pulse_root),
        "device": iv_device_filter,
        "device_arg": args.device,
        "pulse_device": pulse_device_filter,
        "pulse_device_arg": args.pulse_device,
        "input_source": "manual" if args.input_text else "random",
        "random_rows": args.random_rows,
        "random_cols": args.random_cols,
        "random_low": args.random_low,
        "random_high": args.random_high,
        "seed": args.seed,
        "source_files": lib.source_files,
        "source_points": lib.source_points,
        "states": int(lib.states_s.size),
        "pulse_source_files": pulse_lib.source_files,
        "pulse_source_points": pulse_lib.source_points,
        "requested_write_method": args.write_method,
        "write_method_used": write_method_used,
        "write_tolerance_s": args.write_tolerance_us * 1e-6,
        "max_write_pulses": args.max_write_pulses,
        "read_v": args.read_v,
        "tolerance_v": args.tolerance_v,
        "g_min_s": lib.g_min,
        "g_max_s": lib.g_max,
        "dynamic_range": lib.dynamic_range,
        "write_g_min_s": write_lib.g_min,
        "write_g_max_s": write_lib.g_max,
        "write_dynamic_range": write_lib.dynamic_range,
        "mode": args.mode,
        "metrics": metrics,
        "output_dir": str(out_dir),
    }
    write_summary(out_dir / "summary.json", summary)

    np.set_printoptions(precision=4, suppress=True)
    print("\n[Conductance library]")
    print(f"  files: {lib.source_files}, points: {lib.source_points}, states: {lib.states_s.size}")
    print(f"  G range: {lib.g_min * 1e6:.4f} to {lib.g_max * 1e6:.4f} uS")
    print(f"  dynamic range: {lib.dynamic_range:.2f}x")
    print(f"  IV device filter: {iv_device_filter or '(none)'}")
    print("\n[Write simulation]")
    print(f"  method: {write_method_used}")
    print(f"  pulse filter: {pulse_device_filter or '(none)'}")
    print(f"  pulse files: {pulse_lib.source_files}, pulse states: {pulse_lib.source_points}")
    print(f"  write G range: {write_lib.g_min * 1e6:.6f} to {write_lib.g_max * 1e6:.6f} uS")
    if args.write_method == "closed-loop" and write_method_used != "closed-loop":
        print("  closed-loop pulse data not found; fell back to nearest-state mapping")
    print("\n[Input]")
    print(f"  source: {'manual --input' if args.input_text else 'random'}")
    if not args.input_text:
        print(
            f"  shape: {args.random_rows}x{args.random_cols}, "
            f"range: {args.random_low}..{args.random_high}, seed: {args.seed}"
        )
    print(input_matrix)
    print("\n[Kernel]")
    print("  target:")
    print(target_kernel)
    print("  experimental effective:")
    print(experimental_kernel)
    print("\n[Convolution output]")
    print("  theoretical:")
    print(theoretical_output)
    print("  experimental:")
    print(experimental_output)
    print("  error:")
    print(experimental_output - theoretical_output)
    print("\n[Metrics]")
    print(
        f"  MAE={metrics['mae']:.6g}, RMSE={metrics['rmse']:.6g}, "
        f"max_abs_pct={metrics['max_abs_pct']:.3f}%"
    )
    if not PLOT_AVAILABLE:
        print("\n[Plot] matplotlib is not installed; skipped PNG generation.")
    print(f"\n[Saved] {out_dir}")


if __name__ == "__main__":
    main()
