import os
import re
import pandas as pd
from glob import glob
import math
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

# occupancy data and rolling mean per minute
def occupancy_minute_rolling(df: pd.DataFrame, date_str: str, start_hour: int = 6) -> pd.DataFrame:
    """
    Given a raw occupancy DataFrame (for one day) with a 'time' column in seconds (0 at start_hour)
    and an 'N' people-count column sampled every second, compute:
      - datetime = date + start_hour + time(sec)
      - 60-second rolling mean of N
      - value at each full minute (take the last value within each minute)

    Returns columns: ['date', 'datetime', 'N'] where 'datetime' is minute-aligned.
    """
    # datetime = date + start_hour + time(s)
    date = pd.to_datetime(date_str)
    start_offset = pd.to_timedelta(start_hour, unit='h')
    df = df.copy()
    df['date'] = date
    df['datetime'] = date + start_offset + pd.to_timedelta(df['time'], unit='s')

    # Ensure chronological order
    df = df.sort_values('datetime')

    # Per-second series; 60-sample rolling (since data are per second)
    s = df.set_index('datetime')['N']
    r60 = s.rolling(window=60, min_periods=1).mean()

    # Sample once per minute: last value within each minute
    occ_minute = (
        r60.resample('T').last().reset_index()
    )
    occ_minute['date'] = date

    return occ_minute[['date', 'datetime', 'N']]

# steady state model
def steady_state_model(n, G, V, Cs, Cr):
    """
    Compute the air change rate with the steady-state method.

    Parameters:
    n (int): Number of people in steady state
    G (float): CO2 generation rate in L/min
    V (float): Volume of the space (m^3)
    Cs (float): Steady-state CO2 level (ppm)
    Cr (float): Outdoor CO2 level (ppm), default is 400 ppm

    Returns:
    float: Air change rate (1/h)
    """
    return 6 * 10**4 * n * G / (V * (Cs - Cr)) 

# transient mass balance model
def transient_mass_balance_model(A, C, n, V, Cr, G, dt):
    """
    Estimate the air exchange rate using the transient mass balance model.

    Parameters:
    A (float): Air exchange rate (1/h)
    C (float): Current CO2 level (ppm)
    n (int): Number of people in the space
    V (float): Volume of the space (m^3)
    Cr (float): Outdoor CO2 level (ppm)
    G (float): CO2 generation rate per person (L/min)
    dt (float): Timestep (hours)

    Returns:
    float: Predicted CO2 level at the next timestep (ppm)
    """
    Q = A * V
    C1hat = (6 * 10**4 * n * G / Q * (1 - math.exp(-Q / V * dt)) + 
             (C - Cr) * math.exp(-Q / V * dt) + Cr)
    return C1hat

# residual sum of squares
def residual_sum_of_squares(params, C, n, V, G, dt):
    """
    Calculate the residual sum of squares for the transient mass balance model.

    Parameters:
    params (list): List containing A and Cr
    C (list): List of CO2 levels (ppm)
    n (list): List of number of people in the space
    V (float): Volume of the space (m^3)
    G (float): CO2 generation rate per person (L/min)
    dt (float): Timestep (hours)

    Returns:
    float: Residual sum of squares
    """
    A, Cr = params
    rss = 0
    for i in range(1, len(C)):
        C_pred = transient_mass_balance_model(A, C[i-1], n[i], V, Cr, G, dt)
        rss += (C[i] - C_pred) ** 2
    return rss

# optimizer
def optimize_parameters(C, n, V, G, dt, A_init, Cr_init, A_bounds, Cr_bounds):
    """
    Optimize the parameters A and Cr to minimize the residual sum of squares.

    Parameters:
    C (list): List of CO2 levels (ppm)
    n (list): List of number of people in the space
    V (float): Volume of the space (m^3)
    G (float): CO2 generation rate per person (L/min)
    dt (float): Timestep (hours)
    A_init (float): Initial value for A
    Cr_init (float): Initial value for Cr
    A_bounds (tuple): Bounds for A (lower, upper)
    Cr_bounds (tuple): Bounds for Cr (lower, upper)

    Returns:
    dict: Optimized parameters A and Cr
    """
    result = minimize(residual_sum_of_squares, [A_init, Cr_init], args=(C, n, V, G, dt),
                      bounds=[A_bounds, Cr_bounds])
    return {'A': result.x[0], 'Cr': result.x[1]}


# CO2 data
co2_df = pd.read_csv('data-clean/environmental/co2-temp-humidity.csv')
co2_df['datetime'] = pd.to_datetime(co2_df['datetime'])
co2_df['date'] = co2_df['datetime'].dt.date
co2_df['datetime'] = co2_df['datetime'].dt.round('T')
co2_df = co2_df[['device', 'date', 'datetime', 'co2', 'co2_outdoor']]

# occupancy data
occupancy_dir: str = 'data-clean/tracking/occupancy'
paths = sorted(glob(os.path.join(occupancy_dir, '*.csv')))
out = []
for p in paths:
    m = re.search(r'(\d{4}-\d{2}-\d{2})\.csv$', os.path.basename(p))
    date_str = m.group(1)
    df_raw = pd.read_csv(p)
    out.append(occupancy_minute_rolling(df_raw, date_str))
occupancy_df = pd.concat(out, ignore_index=True, copy=False)

# merge on minute-aligned datetime (left join preserves all co2 rows)
df = pd.merge(co2_df, occupancy_df[['datetime', 'N']], on='datetime', how='left')

# initialize a list to store the results
results = []

# group the co2 DataFrame by device and date
grouped_df = df.groupby(['device', 'date'])

# loop through each group (device and date)
for (device, date), group in grouped_df:
    C = group['co2'].tolist()
    Co = group['co2_outdoor'].tolist()[0]
    Co = min(min(C), Co) + 1
    n = group['N'].tolist()
    V = 1178.7  # Volume of waiting area in m^3
    G = 0.004 * 60  # Assumed CO2 generation rate per person in L/min
    dt = 5 / 60  # Timestep in hours (should be 5 minutes)
    
    # ensure C and n are of the same length by taking the lead of C
    C = C[2:]  # Remove the first two values
    n = n[1:-1]  # Remove the first (NA) and last value

    A_init = 5  # Initial value for A
    Cr_init = Co  # Initial value for Cr
    Cr_lower = min(300, Co)
    Cr_upper = min(500, Co)
    A_bounds = (0.1, 100.0)  # Bounds for A
    Cr_bounds = (Cr_lower, Cr_upper)  # Bounds for Cr
    
    # transient mass balance model
    optimized_params = optimize_parameters(C, n, V, G, dt, A_init, Cr_init, A_bounds, Cr_bounds)
    
    # steady state model
    A_ssm = steady_state_model(max(n), G, V, max(C), Co)
    
    # append results
    results.append({'device': device, 'date': date, 'aer_tmb': optimized_params['A'], 'Cr_tmb': optimized_params['Cr'], 'aer_ssm': A_ssm, 'Cr_ssm': Co})

# convert the results to a DataFrame
results_df = pd.DataFrame(results)

# sort the results DataFrame by device and then date
results_df = results_df.sort_values(by=['device', 'date'])

# save the results to a CSV file without index but with header
results_df.to_csv('data-clean/environmental/air-exchange-rate.csv', index=False, header=True)
