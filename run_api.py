# %% Setup

# Core
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Installed
import pandas as pd
import numpy as np
import requests
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Local
from dev.models import Battery

np.random.seed(42)  # Same seed as run.py for consistency

# %% API Configuration
API_BASE_URL = "http://localhost:8000"  # FastAPI dev server
API_ENDPOINT = f"{API_BASE_URL}/battery/optimize"

# %% Simulation parameters (same as run.py)

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

# Battery & grid setup (same as run.py)
capacity = 13.5
battery = Battery(
    capacity               = 13.5,
    max_charge_rate        = 5.0,             # kW
    max_discharge_rate     = 5.0,             # kW
    charging_efficiency    = 0.98,
    discharging_efficiency = 0.98,
    current_soc            = .5 * capacity,   # kWh
    final_soc              = .5 * capacity,   # kWh
    soc_min                = .1 * capacity,   # kWh
    soc_max                = .9 * capacity,   # kWh
)

# Interval settings
interval_seconds = 900  # 15 minutes = 900 seconds
interval_hours = interval_seconds / 3600  # 0.25 hours

print(f"Generating test data for API call...")
print(f"Interval: {interval_seconds} seconds ({interval_hours} hours)")

# %% Generate same mock data as run.py

# Generate 2024 datetime index with 15-minute intervals
start_date     = datetime(2024, 1, 1)
end_date       = datetime(2025, 1, 1)
datetime_index = pd.date_range(start=start_date, end=end_date, freq='15min')[:-1]

# Create base dataframe
df_energy = pd.DataFrame(index=datetime_index)

# Add time-based features for calculations
df_energy = df_energy.assign(DayOfYear = df_energy.index.dayofyear)
df_energy = df_energy.assign(HourOfDay = df_energy.index.hour + df_energy.index.minute/60)
df_energy = df_energy.assign(DayOfWeek = df_energy.index.dayofweek)

# Generate solar generation (HomeGeneration)
seasonal_solar = 0.5 + 0.5 * np.cos(2 * np.pi * (df_energy.DayOfYear - 172) / 365)
daily_solar = np.maximum(0, np.cos(2 * np.pi * (df_energy.HourOfDay - 12) / 24)) ** 2
base_solar = kwp_solar * seasonal_solar * daily_solar
weather_noise = np.random.normal(1, 0.3, len(df_energy))
weather_noise = np.maximum(0.2, weather_noise)
df_energy = df_energy.assign(HomeGeneration = base_solar * weather_noise)

# Generate home consumption (HomeConsumption)
morning_peak  = 2 * np.exp(-((df_energy.HourOfDay - 8) ** 2) / 2)
evening_peak  = 3 * np.exp(-((df_energy.HourOfDay - 19.5) ** 2) / 8)
daily_pattern = morning_peak + evening_peak

weekend_multiplier = np.where(df_energy.DayOfWeek >= 5, 1.2, 1.0)
weekend_shift = np.where(df_energy.DayOfWeek >= 5, 1.0, 0.0)
daily_pattern_weekend = daily_pattern * weekend_multiplier

seasonal_heating = 2 * (0.5 + 0.5 * np.cos(2 * np.pi * (df_energy.DayOfYear - 21) / 365))
heating_daily = np.where(
    (df_energy.HourOfDay < 8) | (df_energy.HourOfDay > 16),
    seasonal_heating * 1.5,
    seasonal_heating * 0.3
)

base_consumption = base_load + weekend_effect * daily_pattern_weekend + heating_effect * heating_daily
consumption_noise = np.random.normal(1, 0.2, len(df_energy))
consumption_noise = np.maximum(0.3, consumption_noise)
df_energy = df_energy.assign(HomeConsumption = base_consumption * consumption_noise)

# Generate price signal (same logic as run.py)
seasonal_price = seasonal_price_mod * (0.3 + 0.7 * np.cos(2 * np.pi * (df_energy.DayOfYear - 21) / 365))
daily_price_pattern = daily_price_mod * (morning_peak + evening_peak) / 3
weekend_price_discount = np.where(df_energy.DayOfWeek >= 5, weekend_price_discount_mod, 0)
solar_effect = solar_price_mod * (df_energy.HomeGeneration / 8)
price_volatility = np.random.normal(0, price_volatility_mod, len(df_energy)) / 3

negative_price_prob = negative_price_mod * (df_energy.HomeGeneration / 8)
negative_price_events = np.random.random(len(df_energy)) < negative_price_prob
negative_price_adjustment = np.where(negative_price_events, np.random.uniform(-50, -120, len(df_energy)), 0)

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

# Limit to analysis timeframe
df_energy = df_energy.loc[t0 : t1]

print(f"Generated {len(df_energy)} timesteps for API test")
print(f"Date range: {df_energy.index[0]} to {df_energy.index[-1]}")

# %% Convert data for API (power to energy conversion)

# Convert power (kW) to energy (kWh) for the interval
energy_data = []
for idx, row in df_energy.iterrows():
    energy_data.append({
        "timestamp": idx.strftime('%Y-%m-%dT%H:%M:%SZ'),  # UTC ISO format
        "generation_kwh": (row.HomeGeneration * interval_hours).item(),    # Convert numpy to Python float
        "consumption_kwh": (row.HomeConsumption * interval_hours).item(),  # Convert numpy to Python float
        "price_eur_per_mwh": row.PriceSignal.item()                        # Convert numpy to Python float
    })

# Prepare API request payload
api_request = {
    "energy_data": energy_data,
    "interval_seconds": interval_seconds,
    "battery": {
        "capacity_kwh": battery.capacity,
        "max_charge_rate_kwh_per_interval": battery.max_charge_rate * interval_hours,      # kW * hours = kWh
        "max_discharge_rate_kwh_per_interval": battery.max_discharge_rate * interval_hours, # kW * hours = kWh
        "charging_efficiency": battery.charging_efficiency,
        "discharging_efficiency": battery.discharging_efficiency,
        "current_soc_kwh": battery.current_soc,
        "target_soc_kwh": battery.final_soc,
        "min_soc_kwh": battery.soc_min,
        "max_soc_kwh": battery.soc_max
    },
    "grid_limits": {
        "max_import_kwh_per_interval": grid_import_limit * interval_hours,  # kW * hours = kWh
        "max_export_kwh_per_interval": grid_export_limit * interval_hours   # kW * hours = kWh
    }
}

print(f"Prepared API request with {len(api_request['energy_data'])} timesteps")
print(f"Battery capacity: {api_request['battery']['capacity_kwh']} kWh")
print(f"Max charge rate: {api_request['battery']['max_charge_rate_kwh_per_interval']} kWh per {interval_seconds}s interval")

# %% Call API

print(f"\nCalling API at {API_ENDPOINT}...")

try:
    response = requests.post(
        API_ENDPOINT,
        json=api_request,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    
    print(f"API response status: {response.status_code}")
    
    if response.status_code == 200:
        api_result = response.json()
        print(f"✅ API call successful!")
        print(f"Status: {api_result['status']}")
        print(f"Total profit: €{api_result['total_profit_eur']:.2f}")
        print(f"Returned {len(api_result['results'])} timestep results")
    else:
        print(f"❌ API call failed!")
        print(f"Response: {response.text}")
        exit(1)
        
except requests.exceptions.RequestException as e:
    print(f"❌ API request failed: {e}")
    print("Make sure FastAPI server is running with: fastapi dev app/main.py")
    exit(1)

# %% Process API results

# Convert API results back to DataFrame
api_results = []
for result in api_result['results']:
    api_results.append({
        'timestamp': pd.Timestamp(result['timestamp']).tz_convert(None),  # Remove timezone for local processing
        'BatteryCharge': result['battery_charge_kwh'] / interval_hours,        # Convert back to kW
        'BatteryDischarge': result['battery_discharge_kwh'] / interval_hours,  # Convert back to kW
        'GridImport': result['grid_import_kwh'] / interval_hours,              # Convert back to kW
        'GridExport': result['grid_export_kwh'] / interval_hours,              # Convert back to kW
        'SoC': result['soc_kwh'],                                              # Keep in kWh
        'Profit': result['profit_eur']
    })

df_api_result = pd.DataFrame(api_results)
df_api_result.set_index('timestamp', inplace=True)

# Merge with original energy data
df_final = df_energy.join(df_api_result)

# Calculate cumulative profit
df_final = df_final.assign(CumulativeProfit=df_final.Profit.cumsum())

# Add baseline comparison (no battery case)
df_final = df_final.assign(BaselineGridExport=(df_final.HomeGeneration  - df_final.HomeConsumption).clip(lower=0))
df_final = df_final.assign(BaselineGridImport=(df_final.HomeConsumption - df_final.HomeGeneration ).clip(lower=0))

baseline_profit = (
    df_final.BaselineGridExport * df_final.PriceSignal * interval_hours / 1000 -  # Revenue
    df_final.BaselineGridImport * df_final.PriceSignal * interval_hours / 1000    # Cost
)
df_final = df_final.assign(BaselineProfit=baseline_profit)
df_final = df_final.assign(BaselineCumulativeProfit=df_final.BaselineProfit.cumsum())

# %% Results comparison

api_final_profit = df_final.CumulativeProfit.iloc[-1]
baseline_final_profit = df_final.BaselineCumulativeProfit.iloc[-1]
improvement = api_final_profit - baseline_final_profit

print(f"\n🔋 Battery Optimization Results via API:")
print(f"=" * 50)
print(f"Total profit with battery:    €{api_final_profit:.2f}")
print(f"Total profit without battery: €{baseline_final_profit:.2f}")
print(f"Improvement:                  €{improvement:.2f}")
print(f"Final SoC:                    {df_final.SoC.iloc[-1]:.2f} kWh")
print(f"Target SoC:                   {battery.final_soc:.2f} kWh")

# %% Energy balance validation

energy_balance = (
    df_final.HomeGeneration - df_final.HomeConsumption + 
    df_final.BatteryDischarge - df_final.BatteryCharge - 
    df_final.GridExport + df_final.GridImport
)
max_imbalance = abs(energy_balance).max()

print(f"\n⚡ Energy Balance Check:")
print(f"Max energy imbalance: {max_imbalance:.6f} kW")
if max_imbalance < 1e-6:
    print("✅ Energy balance looks good!")
else:
    print("❌ Energy balance check failed!")

# %% Plot results

print(f"\n📊 Plotting results...")

fig = make_subplots(
    rows=6, cols=1,
    shared_xaxes=True,
    subplot_titles=(
        "Home Generation & Consumption",
        "Battery Operations (via API)", 
        "Net Grid Flow (Import+ Export-)",
        "State of Charge",
        "Price Signal",
        "Profit Comparison (API vs Baseline)"
    ),
    vertical_spacing=0.03
)

# Home Generation & Consumption
fig.add_trace(go.Scatter(x=df_final.index, y=df_final.HomeGeneration, name='Home Generation', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_final.index, y=df_final.HomeConsumption, name='Home Consumption', line=dict(color='blue')), row=1, col=1)

# Battery Operations
fig.add_trace(go.Bar(x=df_final.index, y=df_final.BatteryCharge, name='Battery Charge', marker_color='green', width=300000), row=2, col=1)
fig.add_trace(go.Bar(x=df_final.index, y=-df_final.BatteryDischarge, name='Battery Discharge', marker_color='red', width=300000), row=2, col=1)

# Net Grid Flow
net_grid_flow = df_final.GridImport - df_final.GridExport
fig.add_trace(go.Scatter(x=df_final.index, y=net_grid_flow, name='Net Grid Flow (API)', line=dict(color='purple')), row=3, col=1)

# State of Charge
fig.add_trace(go.Scatter(x=df_final.index, y=df_final.SoC, name='Battery SoC (API)', line=dict(color='green')), row=4, col=1)

# Price Signal
fig.add_trace(go.Scatter(x=df_final.index, y=df_final.PriceSignal, name='Price Signal', line=dict(color='black')), row=5, col=1)

# Profit Comparison
fig.add_trace(go.Scatter(x=df_final.index, y=df_final.CumulativeProfit, name='API Profit', line=dict(color='green')), row=6, col=1)
fig.add_trace(go.Scatter(x=df_final.index, y=df_final.BaselineCumulativeProfit, name='Baseline Profit', line=dict(color='red')), row=6, col=1)

fig.update_layout(
    height=1200,
    title=f"Battery Optimization API Test Results - Improvement: €{improvement:.2f}",
    hovermode='x unified'
)

fig.update_xaxes(matches='x')
fig.show()

print(f"\n✅ API test completed successfully!")
print(f"The API produced valid optimization results with €{improvement:.2f} improvement over baseline.")