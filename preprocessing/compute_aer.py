import os
import pandas as pd
import math
from scipy.optimize import minimize
from building import *
from concurrent.futures import ProcessPoolExecutor

# co2 data
co2 = pd.read_csv('data-clean/environmental/co2-temp-humidity.csv')
co2['datetime'] = pd.to_datetime(co2['datetime'])
co2['date'] = co2['datetime'].dt.date
co2['datetime'] = co2['datetime'].dt.round('T')
co2 = co2[['device', 'date', 'datetime', 'co2', 'co2_outdoor']]

# occupancy data
occupancy = pd.read_csv('data-clean/tracking/occupancy.csv')
occupancy['time_minute'] = pd.to_datetime(occupancy['time_minute'])
occupancy.rename(columns={'track_id_count': 'no_people'}, inplace=True)
occupancy.rename(columns={'time_minute': 'datetime'}, inplace=True)

# merge co2 and occupancy data
df = pd.merge(co2, occupancy, on='datetime', how='left')

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
    G (float): CO2 generation rate per person (L/h)
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
    G (float): CO2 generation rate per person (L/h)
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
    G (float): CO2 generation rate per person (L/h)
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

# Initialize a list to store the results
results = []

# Group the co2 DataFrame by device and date
grouped_df = df.groupby(['device', 'date'])

# Loop through each group (device and date)
for (device, date), group in grouped_df:
    C = group['co2'].tolist()
    Co = group['co2_outdoor'].tolist()[0]
    Co = min(min(C), Co) + 1
    n = group['no_people'].tolist()
    V = vol  # Volume of waiting room m^3
    G = 0.004 * 60  # Assumed CO2 generation rate per person in L/h
    dt = 5 / 60  # Timestep in hours (should be 5 minutes)
    
    # Ensure C and n are of the same length by taking the lead of C
    C = C[2:]  # Remove the first two values
    n = n[1:-1]  # Remove the first (NA) and last value

    A_init = 1  # Initial value for A
    Cr_init = Co  # Initial value for Cr
    Cr_lower = min(300, Co)
    Cr_upper = min(500, Co)
    A_bounds = (0.1, 100.0)  # Bounds for A
    Cr_bounds = (Cr_lower, Cr_upper)  # Bounds for Cr
    
    # transient mass balance model
    optimized_params = optimize_parameters(C, n, V, G, dt, A_init, Cr_init, A_bounds, Cr_bounds)
    
    # steady state model
    A_ssm = steady_state_model(max(n), G, vol, max(C), Co)
    
    # append results
    results.append({'device': device, 'date': date, 'aer_tmb': optimized_params['A'], 'Cr_tmb': optimized_params['Cr'], 'aer_ssm': A_ssm, 'Cr_ssm': Co})

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Sort the results DataFrame by device and then date
results_df = results_df.sort_values(by=['device', 'date'])

# Save the results to a CSV file without index but with header
results_df.to_csv('data-clean/environmental/air-exchange-rate.csv', index=False, header=True)
