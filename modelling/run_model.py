import os
import pandas as pd
import numpy as np
import pickle
import ast
import argparse
import spatiotemporal_diffusion as spd
import modelling.compute_quanta_exposure as cri
import warnings
import concurrent.futures
from functools import partial
import re

linked_tb_path = 'data-clean/tracking/linked-tb/'
dates = [
    file.replace('.csv', '') 
    for file in os.listdir(linked_tb_path) 
    if file.endswith('.csv')
]
dates = [d for d in dates if d != "2024-06-26"]

def parse_args():
    parser = argparse.ArgumentParser(description="Run risk model simulation.")
    parser.add_argument('--name', type=str, required=True, help='Name of the model run')
    parser.add_argument('--date', type=str, required=True, help='Date of the model run')
    parser.add_argument('--sim', type=str, default='(1,1)', help='Simulation numbers start and end as tuple, e.g. "(1,1)"')
    parser.add_argument('--aer', type=float, default=None, help='Air exchange rate (1/h)')
    parser.add_argument('--inact_rate', type=float, default=None, help='Inactivation rate (1/h)')
    parser.add_argument('--settl_rate', type=float, default=None, help='Settling rate (1/h)')
    parser.add_argument('--quanta_rate', type=str, default=None, help='Quanta rates (1/h) for waiting and walking as a tuple, e.g. "(1.0, 2.0)"')
    parser.add_argument('--breath_rate', type=str, default=None, help='Breathing rates (m3/h)for waiting and walking as a tuple, e.g. "(0.5, 0.6)"')
    parser.add_argument('--cell_size', type=float, default=0.5, help='Size of the cell (m)')
    parser.add_argument('--cell_height', type=float, default=3.0, help='Height of the cell (m)')
    parser.add_argument('--space_vol', type=float, default=None, help='Volume of the space (m3)')
    parser.add_argument('--cores', type=int, default=4, help='Number of cores to use for parallel processing of multiple dates.')
    return parser.parse_args()

def run_for_date(date, args, sim, quanta_rate, breath_rate):
    model_risk(
        name=args.name,
        date=date,
        sim=sim,
        aer=args.aer,
        inact_rate=args.inact_rate,
        settl_rate=args.settl_rate,
        quanta_rate=quanta_rate,
        breath_rate=breath_rate,
        cell_size=args.cell_size,
        cell_height=args.cell_height,
        space_vol=args.space_vol
    )

def model_risk(name, date, sim=(1,1), aer=None, inact_rate=None, settl_rate=None, quanta_rate=None, breath_rate=None, cell_size=0.5, cell_height=3.0, space_vol=None):
    """
    Calculate risk model based on provided parameters.

    Parameters:
        name (str): Name of the model run.
        date (str): Date of the model run. Cannot be empty.
        sim (tuple of int): Simulation start and end numbers. Default (1,1).
        aer (float, optional): Air exchange rate.
        inact_rate (float, optional): Inactivation rate.
        settl_rate (float, optional): Settling rate.
        quanta (tuple of floats, optional): 2D tuple of quanta values for waiting and walking.
        breath_rate (tuple of floats, optional): 2D tuple of breath rate values for waiting and walking.
        cell_size (float): Size of the cell. Cannot be empty.
        cell_height (float): Height of the cell. Cannot be empty.
        space_vol (float, optional): Volume of the space. Cannot be empty.
    """
    print(f"Starting modelling for '{name}' on date '{date}'...")
    # Create the directory to save the results
    results_dir = os.path.join('modelling-results', name, date)
    existing_sims = set()
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        for fname in files:
            existing_sims.update(map(int, re.findall(r'\d{1,5}', fname)))
    existing_sims = list(existing_sims)
    sim_range = set(range(sim[0], sim[1] + 1))
    missing_sims = sorted(sim_range - set(existing_sims))
    if not missing_sims:
        print(f"All simulations in range {sim} already exist for '{name}' on date '{date}'. Skipping.")
        return
    os.makedirs(results_dir, exist_ok=True)
    
    # Check required parameters
    if not date:
        raise ValueError("date cannot be empty")
    if cell_size is None:
        raise ValueError("cell_size cannot be empty")
    if cell_height is None:
        raise ValueError("cell_height cannot be empty")
    
    # Load data about positions of TB and non-TB patients
    tb_pos_path = f'data-clean/tracking/tb-positions/{date}.csv'
    non_tb_ps_path = f'data-clean/tracking/non-tb-positions/{date}.csv'

    if not os.path.exists(tb_pos_path):
        raise FileNotFoundError(f"File not found: {tb_pos_path}")
    if not os.path.exists(non_tb_ps_path):
        raise FileNotFoundError(f"File not found: {non_tb_ps_path}")

    tb_pos_df = pd.read_csv(tb_pos_path)
    non_tb_ps_df = pd.read_csv(non_tb_ps_path)
    
    # Load the building mask
    building_mask_path = 'data-clean/building/building-grid-mask.npy'
    if not os.path.exists(building_mask_path):
        raise FileNotFoundError(f"File not found: {building_mask_path}")
    mask = np.load(building_mask_path)
    active_cells = np.count_nonzero(mask)
    
    # Load the volume of the space if not provided
    if space_vol is None:
        volume_path = 'data-clean/building/building-volume.npy'
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"File not found: {volume_path}")
        space_vol = np.load(volume_path).item()
    
    # Load air exchange rate if not provided
    if aer is None:
        aer_path = 'data-clean/environmental/air-exchange-rate.csv'
        if not os.path.exists(aer_path):
            raise FileNotFoundError(f"File not found: {aer_path}")
        aer_df = pd.read_csv(aer_path)
        aer_row = aer_df[(aer_df['date'] == date) & (aer_df['device'] == 'Aranet4 272D2')]
        if aer_row.empty:
            raise ValueError(f"No air exchange rate found for date {date} and device Aranet4 272D2")
        aer = aer_row.iloc[0]['aer_tmb']
    aer = aer / 3600
    
    # Load quanta rate samples if not provided
    if quanta_rate is None:
        quanta_waiting_path = 'data-clean/assumptions/quanta_waiting.pkl'
        quanta_walking_path = 'data-clean/assumptions/quanta_walking.pkl'
        if not os.path.exists(quanta_waiting_path):
            raise FileNotFoundError(f"File not found: {quanta_waiting_path}")
        if not os.path.exists(quanta_walking_path):
            raise FileNotFoundError(f"File not found: {quanta_walking_path}")
        with open(quanta_waiting_path, 'rb') as f:
            quanta_waiting = pickle.load(f)
        with open(quanta_walking_path, 'rb') as f:
            quanta_walking = pickle.load(f)
    else:
        quanta_rate = (quanta_rate[0] / 3600, quanta_rate[1] / 3600)
        
    # Load inactivation rate samples if not provided
    if inact_rate is None:
        inactivation_path = 'data-clean/assumptions/inactivation.pkl'
        if not os.path.exists(inactivation_path):
            raise FileNotFoundError(f"File not found: {inactivation_path}")
        with open(inactivation_path, 'rb') as f:
            inact_rate = pickle.load(f)
    else:
        inact_rate = [inact_rate / 3600]

    # Load settling rate samples if not provided
    if settl_rate is None:
        settling_path = 'data-clean/assumptions/settling.pkl'
        if not os.path.exists(settling_path):
            raise FileNotFoundError(f"File not found: {settling_path}")
        with open(settling_path, 'rb') as f:
            settl_rate = pickle.load(f)
    else:
        settl_rate = [settl_rate / 3600]
            
    # Load breath rate samples if not provided
    if breath_rate is None:
        breath_rate_path = 'data-clean/assumptions/inhalation_rates.pkl'
        if not os.path.exists(breath_rate_path):
            raise FileNotFoundError(f"File not found: {breath_rate_path}")
        with open(breath_rate_path, 'rb') as f:
            breath_rate = pickle.load(f)
            
    # Compute cell volume
    cell_volume = cell_size * cell_size * cell_height
    
    # Compute diffusion rate
    diffusion_rate = (0.52 * aer + 8.61e-5) * (space_vol**(2/3)) 
    
    # Model quanta concentration and risk
    for sim_num in missing_sims:
        # Quanta generation rate per track and activity
        tb_s = tb_pos_df.copy()
        def replace_activity_with_quanta(row):
            positions = ast.literal_eval(row['positions']) if isinstance(row['positions'], str) else row['positions']
            if not positions:
                return positions
            new_positions = []
            for tup in positions:
                track_id, x, y, activity = tup
                if activity == 0:
                    if quanta_rate is not None:
                        q = quanta_rate[0]
                    else:
                        q = quanta_waiting[track_id][sim_num - 1]
                else:
                    if quanta_rate is not None:
                        q = quanta_rate[1]
                    else:
                        q = quanta_walking[track_id][sim_num - 1]
                new_positions.append((track_id, x, y, q))
            return new_positions
        tb_s['positions'] = tb_s.apply(replace_activity_with_quanta, axis=1)
        
        # Removal rate
        removal_rate = aer + settl_rate[sim_num - 1] + inact_rate[sim_num - 1] 
        
        # Prepare the solver
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver, idx_map, keep = spd.prepare_be_solver(mask, cell_size, diffusion_rate, removal_rate, 1.0)
        
        # Simulate quanta concentration
        quanta_list = []
        quanta = np.zeros_like(mask, dtype=float)
        for idx, row in tb_s.iterrows():
            positions = row['positions']
            positions = [(z, y, r) for (_, y, z, r) in positions] if positions != [] else []
            duration = row['dt']
            _, quanta_new = spd.solve_diffusion_be(quanta, positions, (0, duration), 1, solver, idx_map, keep, mask)
            quanta_list.append(quanta_new[1:])
            quanta = quanta_new[-1]
        quanta = np.concatenate(quanta_list, axis=0)
        
        # Compute risk of infection
        risk_df = pd.DataFrame([
            {
            'new_track_id': tid,
            'hour_idx': h,
            'duration': T,
            'conc_diffusion': Qd,
            'conc_mixed': Qm
            }
            for tid, group in non_tb_ps_df.groupby('new_track_id')
            for (tid, h, T, Qd, Qm) in cri.compute_exposure(
            group, quanta, cell_volume, breath_rate[0], breath_rate[1], active_cells,
            steps_per_hour=3600, dt_seconds=1.0,
            track_id=tid
            )
        ])

        # Save the risk results
        risk_file = os.path.join(results_dir, f'risk_results_sim_{sim_num}.csv')
        risk_df.to_csv(risk_file, index=False)
        print(f"Results for simulation '{name}' on date '{date}' number {sim_num} have been saved to {risk_file}")
    
    print(f"Modelling for '{name}' on date '{date}' has finished.")


if __name__ == "__main__":
    args = parse_args()
    sim = ast.literal_eval(args.sim)
    quanta_rate = ast.literal_eval(args.quanta_rate) if args.quanta_rate is not None else None
    breath_rate = ast.literal_eval(args.breath_rate) if args.breath_rate is not None else None

    if args.date == 'all':
        cores = getattr(args, 'cores', 4)
        partial_func = partial(run_for_date, args=args, sim=sim, quanta_rate=quanta_rate, breath_rate=breath_rate)
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            executor.map(partial_func, dates)
    else:
        model_risk(
            name=args.name,
            date=args.date,
            sim=sim,
            aer=args.aer,
            inact_rate=args.inact_rate,
            settl_rate=args.settl_rate,
            quanta_rate=quanta_rate,
            breath_rate=breath_rate,
            cell_size=args.cell_size,
            cell_height=args.cell_height,
            space_vol=args.space_vol
        )


