# %% Setup

# Core
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Installed
import pandas as pd
import numpy as np
import pulp as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local
from models import Battery

np.random.seed(42)  # The Answer to the Ultimate Question of Life, The Universe, and Everything


# %% Note

"""
This simulation assumes revenue based on time-arbitrage only on EPEX-spot & Tennet intra-day imbalance markets.
In production you would:
- Forecast load & generation instead of assuming you have perfect knowledge.
- Re-run the analysis every time you make new forecasts for any input.
- Unaccounted for revenue streams:
    - Include FCR trading. (Dumping energy asap).
    - Include aFRR trading. (Following setpoints).
    - Include emergency power reserve trading. 
    - Include gains from increasing solar generation capacity.
    - Include gains from reducing grid-connection size.
"""


# %% Simulation parameters

# Timeperiod for analysis
t0 = pd.Timestamp('2024-06-01')
t1 = pd.Timestamp('2024-06-03')

# Power generation & use modifiers
kwp_solar      = 8    # kW
base_load      = 0.5  # kW
weekend_effect = 1.2
heating_effect = 1

# Grid connection
grid_import_limit      = 17.25   # kW (3x25A typical home connection)
grid_export_limit      = 17.25   # kW (same as import for simplicity)

# Price modifiers
base_price                 = 80   # €/MWh
seasonal_price_mod         = 30
daily_price_mod            = 25
weekend_price_discount_mod = -15
solar_price_mod            = -20
price_volatility_mod       = 25
negative_price_mod         = 0.02
final_price_mod            = 1

# Battery & grid setup
capacity = 13.5  # So it can be used repeatedly to set defaults
battery = Battery(
    capacity               = 13.5,            # kWh (typical Tesla Powerwall size)

    # Rate
    max_charge_rate        = 5.0,             # kW
    max_discharge_rate     = 5.0,             # kW

    # Efficiency
    charging_efficiency    = 0.98,
    discharging_efficiency = 0.98,

    # SoC
    current_soc            = .5 * capacity,   # kWh
    final_soc              = .5 * capacity,   # kWh  same as current_soc to discourage dumping charge to game the objective function by the optimizer.
    soc_min                = .1 * capacity,   # kWh
    soc_max                = .9 * capacity,   # kWh
)

# Removed price forecast params - using exact prices instead


# %% Mocking data

# Generate 2024 datetime index with 15-minute intervals
start_date     = datetime(2024, 1, 1)
end_date       = datetime(2025, 1, 1)
datetime_index = pd.date_range(start=start_date, end=end_date, freq='15min')[:-1]  # Exclude 2025-01-01

# Create base dataframe
df_energy = pd.DataFrame(index=datetime_index)

# Add time-based features for calculations
df_energy = df_energy.assign(DayOfYear = df_energy.index.dayofyear)
df_energy = df_energy.assign(HourOfDay = df_energy.index.hour + df_energy.index.minute/60)
df_energy = df_energy.assign(DayOfWeek = df_energy.index.dayofweek)                         # 0=Monday, 6=Sunday

# Generate solar generation (HomeGeneration)

# Seasonal component: higher in summer (cos wave with peak around day 172 = June 21)
seasonal_solar = 0.5 + 0.5 * np.cos(2 * np.pi * (df_energy.DayOfYear - 172) / 365)

# Daily component: sunrise/sunset pattern (cos wave peaking at noon)
daily_solar = np.maximum(0, np.cos(2 * np.pi * (df_energy.HourOfDay - 12) / 24)) ** 2

# Scale to realistic kW values (0-8 kW peak for typical home solar)
base_solar = kwp_solar * seasonal_solar * daily_solar

# Add weather noise (some days are cloudier)
weather_noise = np.random.normal(1, 0.3, len(df_energy))
weather_noise = np.maximum(0.2, weather_noise)            # Minimum 20% of clear sky (ensures some daily generation)

df_energy = df_energy.assign(HomeGeneration = base_solar * weather_noise)

# Generate home consumption (HomeConsumption)

# Daily pattern: higher in morning (7-9) and evening (17-22)
morning_peak  = 2 * np.exp(-((df_energy.HourOfDay - 8) ** 2) / 2)
evening_peak  = 3 * np.exp(-((df_energy.HourOfDay - 19.5) ** 2) / 8)
daily_pattern = morning_peak + evening_peak

# Weekend effect: slightly higher consumption, shifted patterns
weekend_multiplier    = np.where(df_energy.DayOfWeek >= 5, 1.2, 1.0)
weekend_shift         = np.where(df_energy.DayOfWeek >= 5, 1.0, 0.0)  # Later wake-up on weekends
daily_pattern_weekend = daily_pattern * weekend_multiplier

# Seasonal heating (higher consumption in winter)
seasonal_heating = 2 * (0.5 + 0.5 * np.cos(2 * np.pi * (df_energy.DayOfYear - 21) / 365))  # Peak around Jan 21

# Temperature-dependent heating (kicks in more during cold hours)
heating_daily = np.where(
    (df_energy.HourOfDay < 8) | (df_energy.HourOfDay > 16),
    seasonal_heating * 1.5,
    seasonal_heating * 0.3
)

# Combine all consumption components
base_consumption = base_load + weekend_effect * daily_pattern_weekend + heating_effect * heating_daily

# Add consumption noise
consumption_noise = np.random.normal(1, 0.2, len(df_energy))
consumption_noise = np.maximum(0.3, consumption_noise)

df_energy = df_energy.assign(HomeConsumption = base_consumption * consumption_noise)

# Generate price signal (PriceSignal) in €/MWh

# Seasonal component: higher in winter due to heating demand
seasonal_price = seasonal_price_mod * (0.3 + 0.7 * np.cos(2 * np.pi * (df_energy.DayOfYear - 21) / 365))

# Daily pattern: higher during peak hours (morning/evening)
daily_price_pattern = daily_price_mod * (morning_peak + evening_peak) / 3

# Weekend effect: generally lower prices
weekend_price_discount = np.where(df_energy.DayOfWeek >= 5, weekend_price_discount_mod, 0)

# Solar correlation: negative correlation with solar generation (more solar = lower prices)
solar_effect = solar_price_mod * (df_energy.HomeGeneration / 8)  # Normalize to max generation

# Add price volatility and occasional negative prices
price_volatility = np.random.normal(0, price_volatility_mod, len(df_energy)) / 3

# Occasional negative price events (especially during high solar periods)
negative_price_prob = negative_price_mod * (df_energy.HomeGeneration / 8)  # Higher prob during high solar
negative_price_events     = np.random.random(len(df_energy)) < negative_price_prob
negative_price_adjustment = np.where(negative_price_events, np.random.uniform(-50, -120, len(df_energy)), 0)

# Combine all price components
total_price = (
    base_price + 
    seasonal_price + 
    daily_price_pattern + 
    weekend_price_discount + 
    solar_effect + 
    price_volatility + 
    negative_price_adjustment
)

df_energy = df_energy.assign(PriceSignal = total_price * final_price_mod)

print(f"Generated {len(df_energy)} data points for 2024")
print(f"Date range: {df_energy.index[0]} to {df_energy.index[-1]}")
print(f"\nDataFrame columns: {list(df_energy.columns)}")
print(f"\nSample data preview:")
print(df_energy.head())
print(f"\nPrice signal stats (€/MWh):")
print(f"Min: {df_energy.PriceSignal.min():.1f}")
print(f"Max: {df_energy.PriceSignal.max():.1f}")  
print(f"Mean: {df_energy.PriceSignal.mean():.1f}")
print(f"Negative prices: {(df_energy.PriceSignal < 0).sum()} events")


# Removed price forecasting - using exact prices instead
# In dev environment we assume perfect price knowledge
df_energy = df_energy.assign(PriceForecast=df_energy.PriceSignal)

print()
print(f"Using exact prices instead of forecast (dev environment assumption)")
print(f"Price range: {df_energy.PriceSignal.min():.1f} to {df_energy.PriceSignal.max():.1f} €/MWh")


# %% Optimization

# Limit input data to specified time-range
df_energy = df_energy.loc[t0 : t1]

# Convert interval to hours for energy calculations
interval_hours = .25

# Number of timesteps
n_steps   = len(df_energy)
timesteps = range(n_steps)

# Create optimization problem
prob = pl.LpProblem("Battery_Optimization", pl.LpMaximize)

# Decision variables
battery_charge = pl.LpVariable.dicts(
    "BatteryCharge",
    timesteps,
    lowBound=0,
    upBound=battery.max_charge_rate
)
battery_discharge = pl.LpVariable.dicts(
    "BatteryDischarge",
    timesteps,
    lowBound=0,
    upBound=battery.max_discharge_rate
)
grid_import = pl.LpVariable.dicts(
    "GridImport",
    timesteps,
    lowBound=0,
    upBound=grid_import_limit
)
grid_export = pl.LpVariable.dicts(
    "GridExport",
    timesteps,
    lowBound=0,
    upBound=grid_export_limit
)
soc = pl.LpVariable.dicts(
    "SoC",
    timesteps,
    lowBound=battery.soc_min,
    upBound=battery.soc_max
)

# Objective: maximize profit
price_per_kwh = df_energy.PriceForecast / 1000
profit = pl.lpSum([
    grid_export[t] * price_per_kwh.iloc[t] * interval_hours -  # Revenue from export
    grid_import[t] * price_per_kwh.iloc[t] * interval_hours    # Cost of import
    for t in timesteps
])
prob += profit

# Constraints
for t in timesteps:
    # Power balance constraint
    prob += (
        df_energy.HomeGeneration.iloc[t] +
        battery_discharge[t] + grid_import[t]
        ==
        df_energy.HomeConsumption.iloc[t] +
        battery_charge[t] + grid_export[t]
    )

    # SoC dynamics with efficiency losses
    if t == 0:
        # Initial SoC
        prob += soc[t] == (
            battery.current_soc +
            battery_charge[t] * battery.charging_efficiency * interval_hours -
            battery_discharge[t] / battery.discharging_efficiency * interval_hours
        )
    else:
        # SoC evolution
        prob += soc[t] == (
            soc[t-1] +
            battery_charge[t] * battery.charging_efficiency * interval_hours -
            battery_discharge[t] / battery.discharging_efficiency * interval_hours
        )

# Final SoC constraint
prob += soc[n_steps - 1] == battery.final_soc

# Solve the problem
prob.solve(pl.PULP_CBC_CMD(msg=0))  # msg = 0 suppresses solver output

# Check if solution is optimal
if prob.status != pl.LpStatusOptimal:
    raise ValueError(f"Optimization failed with status: {pl.LpStatus[prob.status]}")

# Extract results and add to dataframe
df_result = df_energy.copy()
df_result = df_result.assign(
    BatteryCharge    = [battery_charge[t].varValue for t in timesteps],
    BatteryDischarge = [battery_discharge[t].varValue for t in timesteps],
    GridImport       = [grid_import[t].varValue for t in timesteps],
    GridExport       = [grid_export[t].varValue for t in timesteps],
    SoC              = [soc[t].varValue for t in timesteps]
)

# Add sanity check baseline for no-battery case
df_result = df_result.assign(BaselineGridExport=(df_result.HomeGeneration  - df_result.HomeConsumption).clip(lower=0))
df_result = df_result.assign(BaselineGridImport=(df_result.HomeConsumption - df_result.HomeGeneration ).clip(lower=0))

baseline_price_export = df_result.BaselineGridExport * df_result.PriceSignal * interval_hours / 1000
baseline_price_import = df_result.BaselineGridImport * df_result.PriceSignal * interval_hours / 1000
df_result = df_result.assign(BaselineProfit=baseline_price_export - baseline_price_import)

df_result = df_result.assign(BaselineCumulativeProfit=df_result.BaselineProfit.cumsum())

battery_price_export = df_result.GridExport * df_result.PriceSignal * interval_hours / 1000
battery_price_import = df_result.GridImport * df_result.PriceSignal * interval_hours / 1000
df_result = df_result.assign(BatteryProfit=battery_price_export - battery_price_import)

df_result = df_result.assign(BatteryCumulativeProfit=df_result.BatteryProfit.cumsum())

# Print results
battery_final  = df_result.BatteryCumulativeProfit .iloc[-1]
baseline_final = df_result.BaselineCumulativeProfit.iloc[-1]
print(f"Optimization completed successfully")
print()
print(f"Total profit with battery:               €{battery_final:.2f}")
print(f"Total profit without battery (baseline): €{baseline_final:.2f}")
print(f"Difference:                              €{(battery_final - baseline_final):.2f}")
print()
print(f"Final SoC: {df_result.SoC.iloc[-1]:.2f} kWh (target: {battery.final_soc} kWh)")


# %% Show results

print('Plotting...')
fig = make_subplots(
    rows=10, cols=1,  # Changed from 9 to 10 rows
    shared_xaxes=True,
    subplot_titles=(
        "Home Generation & Consumption",
        "Battery Operations", 
        "Net Grid Flow (Import+ Export-)",
        "State of Charge",
        "Price Signal (Exact)",
        "Price Signal Detail"
        "Profit Comparison",
        "Baseline vs Optimized Net Grid",
        "Cumulative Profit",
        "Energy Balance Check (Should be ~0 at every timestep)",
    ),
    vertical_spacing=0.02
)

# Home Generation & Consumption
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.HomeGeneration, name='Home Generation'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.HomeConsumption, name='Home Consumption'), row=1, col=1)

# Battery Operations
fig.add_trace(go.Bar(x=df_result.index, y=df_result.BatteryCharge, name='Battery Charge', width=300000), row=2, col=1)
fig.add_trace(go.Bar(x=df_result.index, y=-df_result.BatteryDischarge, name='Battery Discharge', width=300000), row=2, col=1)

# Net Grid Flow with battery operation highlighting
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.GridImport - df_result.GridExport, name='Net Grid Flow'), row=3, col=1)

max_opacity = 0.3

# Add individual time interval highlighting based on battery operations
for i, timestamp in enumerate(df_result.index):
    charge_rate = df_result.BatteryCharge.iloc[i]
    discharge_rate = df_result.BatteryDischarge.iloc[i]
    
    # Calculate next timestamp for rectangle width
    if i < len(df_result.index) - 1:
        next_timestamp = df_result.index[i + 1]
    else:
        # For last interval, use same duration as previous
        duration = df_result.index[i] - df_result.index[i-1]
        next_timestamp = timestamp + duration
    
    # Charging periods (green)
    if charge_rate > 0.01:  # threshold to avoid noise
        opacity = min(max_opacity, (charge_rate / battery.max_charge_rate) * max_opacity)
        fig.add_vrect(
            x0=timestamp, x1=next_timestamp,
            fillcolor="Green", opacity=opacity,
            layer="below", line_width=0,
            row=3, col=1
        )
    
    # Discharging periods (red)  
    elif discharge_rate > 0.01:
        opacity = min(max_opacity, (discharge_rate / battery.max_discharge_rate) * max_opacity)
        fig.add_vrect(
            x0=timestamp, x1=next_timestamp,
            fillcolor="Red", opacity=opacity,
            layer="below", line_width=0,
            row=3, col=1
        )

# State of Charge
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.SoC, name='Battery SoC'), row=4, col=1)

# Price Signal (using exact prices)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.PriceSignal, name='Price Signal', line=dict(color='blue')), row=5, col=1)

# Price Signal Detail (same as above, for visibility)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.PriceSignal, name='Price Detail', line=dict(color='green', width=1)), row=6, col=1)

# Profit Comparison
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.BaselineProfit, name='Baseline Profit'), row=7, col=1)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.BatteryProfit, name='Battery Profit'), row=7, col=1)

# Baseline vs Optimized Net Grid
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.BaselineGridImport - df_result.BaselineGridExport, name='Baseline Net Grid'), row=8, col=1)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.GridImport - df_result.GridExport, name='Optimized Net Grid'), row=8, col=1)

# Cumulative Profit
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.BaselineCumulativeProfit, name='Baseline Cumulative'), row=9, col=1)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.BatteryCumulativeProfit, name='Battery Cumulative'), row=9, col=1)

# Energy Balance Check
energy_balance = df_result.HomeGeneration - df_result.HomeConsumption + df_result.BatteryDischarge - df_result.BatteryCharge - df_result.GridExport + df_result.GridImport
fig.add_trace(go.Scatter(x=df_result.index, y=energy_balance, name='Energy Balance'), row=10, col=1)
fig.update_yaxes(range=[-1, 1], row=10, col=1)

fig.update_layout(
    height=2200,  # Increased height for extra subplot
    title="Battery Optimization Physics Check (Using Exact Prices)",
    hovermode='x unified'
)

fig.update_xaxes(matches='x')

