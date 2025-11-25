import os
import re
import ast
import argparse
import warnings
from itertools import repeat
import concurrent.futures as cf
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import spatiotemporal_diffusion as spd
import compute_quanta_exposure as cri


# =============================================================================
# Paths & constants
# =============================================================================

PATH_TRACKING_TB = "data-clean/tracking/tb-positions"
PATH_TRACKING_NONTB = "data-clean/tracking/non-tb-positions"
PATH_LINKED_TB = "data-clean/tracking/linked-tb"

PATH_BUILDING_MASK = "data-clean/building/building-grid-mask.npy"
PATH_BUILDING_VOLUME = "data-clean/building/building-volume.npy"
PATH_BUILDING_CELL_SIZE = "data-clean/building/building-grid-cell-size.npy"
PATH_BUILDING_HEIGHT = "data-clean/building/building-height.npy"

PATH_AER = "data-clean/environmental/air-exchange-rate.csv"
AER_DEVICE = "Aranet4 272D2"  # device to select for AER

PATH_QGEN = "data-clean/assumptions/quanta_generation_rates.csv"
PATH_QREM = "data-clean/assumptions/quanta_removal_rates.csv"

RESULTS_ROOT = "modelling-results"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run risk model simulations.")
    p.add_argument("--name", required=True,
                   help="Folder under modelling-results/ to save outputs (e.g., 'baseline').")
    p.add_argument("--date", required=True,
                   help='"YYYY-MM-DD" to run one date, or "all" to run all available dates.')
    p.add_argument("--sim", default="(1,1)",
                   help='Simulation range as a tuple string, e.g. "(1,100)".')
    p.add_argument("--aer", type=float, default=None,
                   help="Air exchange rate (1/h). If omitted, read from data.")
    p.add_argument("--inact_rate", type=float, default=None,
                   help="Pathogen inactivation rate (1/h). If omitted, use sampled list.")
    p.add_argument("--settl_rate", type=float, default=None,
                   help="Gravitational settling rate (1/h). If omitted, use sampled list.")
    p.add_argument("--quanta_rate", default=None,
                   help='Override quanta (1/h) for infectious==1 by activity (waiting, walking),'
                        ' e.g. "(1.0, 3.0)". Others remain from CSV.')
    p.add_argument("--quanta_mult", default=None,
                   help='Multipliers (HIV+, confirmed TB, presumptive TB), e.g. "(1,1,1)".')
    p.add_argument("--breath_rate", default=None,
                   help='Breathing rates (m3/h) by activity (waiting, walking), e.g. "(0.5, 1.2)".')
    p.add_argument("--cores", type=int, default=4,
                   help="Number of worker processes for parallel runs.")
    return p.parse_args()


# =============================================================================
# Shared data (loaded once)
# =============================================================================

def load_shared_data() -> Dict[str, object]:
    """
    Load heavy/shared inputs once:
      - building mask, geometry
      - quanta generation master table
      - AER table
      - quanta removal samples (inact/settling)
    Returns a dict with consistent, descriptive keys.
    """
    shared: Dict[str, object] = {}

    # Building geometry
    if not os.path.exists(PATH_BUILDING_MASK):
        raise FileNotFoundError(PATH_BUILDING_MASK)
    mask_arr = np.load(PATH_BUILDING_MASK)
    shared["mask_arr"] = mask_arr
    shared["active_cells"] = int(np.count_nonzero(mask_arr))

    if not os.path.exists(PATH_BUILDING_VOLUME):
        raise FileNotFoundError(PATH_BUILDING_VOLUME)
    shared["space_vol"] = float(np.load(PATH_BUILDING_VOLUME).item())

    if not os.path.exists(PATH_BUILDING_CELL_SIZE):
        raise FileNotFoundError(PATH_BUILDING_CELL_SIZE)
    shared["cell_size"] = float(np.load(PATH_BUILDING_CELL_SIZE).item())

    if not os.path.exists(PATH_BUILDING_HEIGHT):
        raise FileNotFoundError(PATH_BUILDING_HEIGHT)
    shared["cell_height"] = float(np.load(PATH_BUILDING_HEIGHT).item())

    shared["cell_vol"] = shared["cell_size"] * shared["cell_size"] * shared["cell_height"]

    # Assumptions (tables loaded once; subset per date later)
    if not os.path.exists(PATH_QGEN):
        raise FileNotFoundError(PATH_QGEN)
    shared["qgen_df"] = pd.read_csv(PATH_QGEN)

    if not os.path.exists(PATH_AER):
        raise FileNotFoundError(PATH_AER)
    shared["aer_df"] = pd.read_csv(PATH_AER)

    if not os.path.exists(PATH_QREM):
        raise FileNotFoundError(PATH_QREM)
    qrem_df = pd.read_csv(PATH_QREM).sort_values("sample")
    shared["inact_list"] = (qrem_df["inactivation"].values / 3600.0).tolist()  # per second
    shared["settl_list"] = (qrem_df["settling"].values / 3600.0).tolist()      # per second

    return shared


# =============================================================================
# Utilities
# =============================================================================

def list_available_dates() -> List[str]:
    """Return available dates (YYYY-MM-DD) inferred from filenames in linked TB folder."""
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(PATH_LINKED_TB)
        if f.endswith(".csv")
    ]

def ensure_dir_exists(path: str) -> str:
    """Create directory if needed and return the path."""
    os.makedirs(path, exist_ok=True)
    return path

def missing_simulations(results_dir: str, sim_range: Tuple[int, int]) -> List[int]:
    """
    Determine which simulations in sim_range are not yet present in results_dir.
    Expects files to contain simulation numbers in their names.
    """
    ensure_dir_exists(results_dir)
    existing: set[int] = set()
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv"):
            existing |= set(map(int, re.findall(r"\d{1,7}", fname)))
    start, end = sim_range
    wanted = set(range(start, end + 1))
    return sorted(wanted - existing)


# =============================================================================
# Per-date preparation (lightweight)
# =============================================================================

def prepare_date_inputs(
    date_str: str,
    args: argparse.Namespace,
    shared: Dict[str, object],
) -> Dict[str, object]:
    """
    Prepare in-memory inputs for a single date:
      - TB and non-TB positions
      - Per-second AER (from override or table)
      - Quanta generation df for the date (per-second, multipliers applied)
      - Per-simulation removal rates list (AER + inact + settl), element-wise
      - Breathing rates (m3/s)
      - Diffusion rate (per second)
    """
    # Positions
    tb_csv = os.path.join(PATH_TRACKING_TB, f"{date_str}.csv")
    nontb_csv = os.path.join(PATH_TRACKING_NONTB, f"{date_str}.csv")
    if not os.path.exists(tb_csv):
        raise FileNotFoundError(tb_csv)
    if not os.path.exists(nontb_csv):
        raise FileNotFoundError(nontb_csv)
    tb_pos_df = pd.read_csv(tb_csv)
    non_tb_df = pd.read_csv(nontb_csv)

    # AER (per second)
    if args.aer is not None:
        aer_per_s = float(args.aer) / 3600.0
    else:
        aer_row = shared["aer_df"][
            (shared["aer_df"]["date"] == date_str) &
            (shared["aer_df"]["device"] == AER_DEVICE)
        ]
        if aer_row.empty:
            raise ValueError(f"No AER for date {date_str} & device {AER_DEVICE}")
        aer_per_s = float(aer_row.iloc[0]["aer_tmb"]) / 3600.0

    # Quanta generation (subset by date; convert to per-second; apply multipliers)
    qgen_df = shared["qgen_df"]
    qgen_date_df = qgen_df[qgen_df["date"] == date_str].copy()
    if qgen_date_df.empty:
        raise ValueError(f"No quanta_generation rows for date {date_str}")

    # Override waiting/walking for infectious==1 OR convert to per-second
    if args.quanta_rate is not None:
        q_wait_h, q_walk_h = ast.literal_eval(args.quanta_rate)
        qgen_date_df.loc[qgen_date_df["infectious"] == 1, "waiting_rate"] = float(q_wait_h) / 3600.0
        qgen_date_df.loc[qgen_date_df["infectious"] == 1, "walking_rate"] = float(q_walk_h) / 3600.0
    else:
        qgen_date_df["waiting_rate"] = qgen_date_df["waiting_rate"] / 3600.0
        qgen_date_df["walking_rate"] = qgen_date_df["walking_rate"] / 3600.0

    # Multipliers
    quanta_mult = (1.0, 1.0, 1.0) if args.quanta_mult is None else tuple(map(float, ast.literal_eval(args.quanta_mult)))
    qgen_date_df["quanta_multiplier"] = 1.0
    qgen_date_df.loc[qgen_date_df["hiv"] == 1, "quanta_multiplier"] *= quanta_mult[0]
    qgen_date_df.loc[qgen_date_df["tb"] == 1,  "quanta_multiplier"] *= quanta_mult[1]
    qgen_date_df.loc[qgen_date_df["tb"] == 0,  "quanta_multiplier"] *= quanta_mult[2]
    qgen_date_df["waiting_rate"] *= qgen_date_df["quanta_multiplier"]
    qgen_date_df["walking_rate"] *= qgen_date_df["quanta_multiplier"]

    # Removal components (per second), per-simulation list
    sim_end = ast.literal_eval(args.sim)[1]
    n_sims = int(sim_end)

    aer_list = [aer_per_s] * n_sims
    if args.inact_rate is not None:
        inact_list = [float(args.inact_rate) / 3600.0] * n_sims
    else:
        inact_list = shared["inact_list"][:n_sims]

    if args.settl_rate is not None:
        settl_list = [float(args.settl_rate) / 3600.0] * n_sims
    else:
        settl_list = shared["settl_list"][:n_sims]

    # IMPORTANT: element-wise sum to get per-sim total removal rate
    removal_rate_list = [
        a + i + s for a, i, s in zip(aer_list, inact_list, settl_list)
    ]

    # Breathing rates (m3/s)
    if args.breath_rate is None:
        br_wait = (0.5 * 0.4632 + 0.5 * 0.5580) / 3600.0
        br_walk = (0.5 * 1.2192 + 0.5 * 1.4478) / 3600.0
        breath_rate = (br_wait, br_walk)
    else:
        br_wait_h, br_walk_h = map(float, ast.literal_eval(args.breath_rate))
        breath_rate = (br_wait_h / 3600.0, br_walk_h / 3600.0)

    # Diffusion rate (per second)
    space_vol = float(shared["space_vol"])
    diffusion_rate = (0.52 * aer_per_s + 8.61e-5) * (space_vol ** (2 / 3))

    return dict(
        date=date_str,
        tb_pos_df=tb_pos_df,
        non_tb_df=non_tb_df,
        qgen_date_df=qgen_date_df,
        removal_rate_list=removal_rate_list,
        breath_rate=breath_rate,
        diffusion_rate=diffusion_rate,
        cell_size=float(shared["cell_size"]),
        cell_vol=float(shared["cell_vol"]),
    )


# =============================================================================
# Single-simulation worker
# =============================================================================

def run_single_sim(sim_num: int, run_name: str, shared: Dict[str, object], din: Dict[str, object]) -> None:
    """
    Run exactly one simulation for a prepared date input `din` and save results.
    """
    date_str = din["date"]
    results_dir = ensure_dir_exists(os.path.join(RESULTS_ROOT, run_name, date_str))

    # Quanta generation for this simulation
    qgen_sim_df = din["qgen_date_df"][din["qgen_date_df"]["sample"] == sim_num]
    if qgen_sim_df.empty:
        raise ValueError(f"No quanta_generation rows for date {date_str}, sim {sim_num}")

    q_wait_map = dict(zip(qgen_sim_df["new_track_id"], qgen_sim_df["waiting_rate"]))
    q_walk_map = dict(zip(qgen_sim_df["new_track_id"], qgen_sim_df["walking_rate"]))

    # Replace activity flag with per-second quanta value
    tb_df = din["tb_pos_df"].copy()

    def _positions_with_quanta(row):
        pos = row["positions"]
        if isinstance(pos, str):
            pos = ast.literal_eval(pos)
        if not pos:
            return pos
        out = []
        for (track_id, x_i, y_k, is_walk) in pos:
            if track_id not in q_wait_map or track_id not in q_walk_map:
                raise ValueError(f"Missing quanta for track_id={track_id} (sim {sim_num})")
            q_val = q_walk_map[track_id] if is_walk == 1 else q_wait_map[track_id]
            out.append((track_id, x_i, y_k, q_val))
        return out

    tb_df["positions"] = tb_df.apply(_positions_with_quanta, axis=1)

    # Prepare solver
    removal_rate = din["removal_rate_list"][sim_num - 1]  # per second
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver, idx_map, keep = spd.prepare_be_solver(
            shared["mask_arr"],
            din["cell_size"],
            din["diffusion_rate"],
            removal_rate,
            1.0,
        )

    # Simulate concentration over time
    quanta_snapshots: List[np.ndarray] = []
    quanta_grid = np.zeros_like(shared["mask_arr"], dtype=float)

    for _, row in tb_df.iterrows():
        pos = row["positions"]
        pos = [(y_k, x_i, q_val) for (_, x_i, y_k, q_val) in pos] if pos else []
        dur = int(row["dt"])
        _, q_series = spd.solve_diffusion_be(
            quanta_grid, pos, (0, dur), 1, solver, idx_map, keep, shared["mask_arr"]
        )
        quanta_snapshots.append(q_series[1:])
        quanta_grid = q_series[-1]

    quanta_time = (
        np.concatenate(quanta_snapshots, axis=0)
        if quanta_snapshots
        else np.zeros((0,) + shared["mask_arr"].shape, dtype=float)
    )

    # Compute risk for non-TB tracks
    br_wait, br_walk = din["breath_rate"]
    rows: List[Dict[str, object]] = []
    for track_id, grp_df in din["non_tb_df"].groupby("new_track_id"):
        for (tid2, hour_idx, duration_s, conc_diff, conc_mixed) in cri.compute_exposure(
            grp_df,
            quanta_time,
            din["cell_vol"],
            br_wait,
            br_walk,
            shared["active_cells"],
            steps_per_hour=3600,
            dt_seconds=1.0,
            track_id=track_id,
        ):
            rows.append({
                "new_track_id": tid2,
                "hour_idx": hour_idx,
                "duration": duration_s,
                "conc_diffusion": conc_diff,
                "conc_mixed": conc_mixed,
            })

    risk_df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, f"risk_results_sim_{sim_num}.csv")
    risk_df.to_csv(out_csv, index=False)
    print(f"[{date_str}] sim {sim_num} -> {out_csv}")


# =============================================================================
# Orchestration
# =============================================================================

def date_worker(date_str: str, args: argparse.Namespace, shared: Dict[str, object], sim_range: Tuple[int, int]) -> None:
    """Run all missing simulations for one date (sequential within a date)."""
    din = prepare_date_inputs(date_str, args, shared)
    out_dir = os.path.join(RESULTS_ROOT, args.name, date_str)
    missing = missing_simulations(out_dir, sim_range)
    for sim_num in missing:
        run_single_sim(sim_num, args.name, shared, din)

def sim_worker(sim_num: int, run_name: str, shared: Dict[str, object], din: Dict[str, object]) -> None:
    """Wrapper to run one simulation (top-level for pickling in ProcessPool)."""
    run_single_sim(sim_num, run_name, shared, din)

def run_single_date(args: argparse.Namespace, shared: Dict[str, object], date_str: str, sim_range: Tuple[int, int]) -> None:
    """Parallelize simulations for one date."""
    din = prepare_date_inputs(date_str, args, shared)
    out_dir = os.path.join(RESULTS_ROOT, args.name, date_str)
    missing = missing_simulations(out_dir, sim_range)
    if not missing:
        print(f"[{date_str}] all simulations {sim_range} already exist.")
        return

    with cf.ProcessPoolExecutor(max_workers=args.cores) as ex:
        list(ex.map(
            sim_worker,
            missing,
            repeat(args.name),
            repeat(shared),
            repeat(din),
        ))

def run_all_dates(args: argparse.Namespace, shared: Dict[str, object], sim_range: Tuple[int, int]) -> None:
    """Parallelize across dates; run simulations sequentially within each date."""
    dates_list = list_available_dates()
    with cf.ProcessPoolExecutor(max_workers=args.cores) as ex:
        list(ex.map(
            date_worker,
            dates_list,
            repeat(args),
            repeat(shared),
            repeat(sim_range),
        ))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    sim_range = ast.literal_eval(args.sim)  # (start, end)

    SHARED = load_shared_data()

    if args.date == "all":
        run_all_dates(args, SHARED, sim_range)
    else:
        run_single_date(args, SHARED, args.date, sim_range)
