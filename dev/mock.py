import pandas as pd
import numpy as np
from datetime import datetime
from models import Battery


"""
The only goal of this file is to setup a dataset that is close enough to reality that it enables testing downstream logic.
Energy data is filled with errors, messy, often incomplete, on top of that its put behind a walled garden. Therefore
the choice was made to utilize a stateless bring-your-own data approach, freeing me up to work on the logic.
"""

# Set random seed for reproducibility
np.random.seed(42)  # The Answer to the Ultimate Question of Life, The Universe, and Everything

# Generate 2024 datetime index with 15-minute intervals
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 1, 1)
datetime_index = pd.date_range(start=start_date, end=end_date, freq='15min')[:-1]  # Exclude 2025-01-01

# Create base dataframe
df_energy = pd.DataFrame(index=datetime_index)

# Add time-based features for calculations
df_energy = df_energy.assign(DayOfYear = df_energy.index.dayofyear)
df_energy = df_energy.assign(HourOfDay = df_energy.index.hour + df_energy.index.minute/60)
df_energy = df_energy.assign(DayOfWeek = df_energy.index.dayofweek)  # 0=Monday, 6=Sunday

# Generate solar generation (HomeGeneration)
# Seasonal component: higher in summer (cos wave with peak around day 172 = June 21)
seasonal_solar = 0.5 + 0.5 * np.cos(2 * np.pi * (df_energy.DayOfYear - 172) / 365)

# Daily component: sunrise/sunset pattern (cos wave peaking at noon)
daily_solar = np.maximum(0, np.cos(2 * np.pi * (df_energy.HourOfDay - 12) / 24)) ** 2

# Scale to realistic kW values (0-8 kW peak for typical home solar)
base_solar = 8 * seasonal_solar * daily_solar

# Add weather noise (some days are cloudier)
weather_noise = np.random.normal(1, 0.3, len(df_energy))
weather_noise = np.maximum(0.2, weather_noise)  # Minimum 20% of clear sky (ensures some daily generation)

df_energy = df_energy.assign(HomeGeneration = base_solar * weather_noise)

# Generate home consumption (HomeConsumption)
# Base load (always-on devices)
base_load = 0.5  # kW

# Daily pattern: higher in morning (7-9) and evening (17-22)
morning_peak = 2 * np.exp(-((df_energy.HourOfDay - 8) ** 2) / 2)
evening_peak = 3 * np.exp(-((df_energy.HourOfDay - 19.5) ** 2) / 8)
daily_pattern = morning_peak + evening_peak

# Weekend effect: slightly higher consumption, shifted patterns
weekend_multiplier = np.where(df_energy.DayOfWeek >= 5, 1.2, 1.0)
weekend_shift = np.where(df_energy.DayOfWeek >= 5, 1.0, 0.0)  # Later wake-up on weekends
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
base_consumption = base_load + daily_pattern_weekend + heating_daily

# Add consumption noise
consumption_noise = np.random.normal(1, 0.2, len(df_energy))
consumption_noise = np.maximum(0.3, consumption_noise)

df_energy = df_energy.assign(HomeConsumption = base_consumption * consumption_noise)

# Generate price signal (PriceSignal) in €/MWh
# Base price level
base_price = 80  # €/MWh

# Seasonal component: higher in winter due to heating demand
seasonal_price = 30 * (0.3 + 0.7 * np.cos(2 * np.pi * (df_energy.DayOfYear - 21) / 365))

# Daily pattern: higher during peak hours (morning/evening)
daily_price_pattern = 25 * (morning_peak + evening_peak) / 3

# Weekend effect: generally lower prices
weekend_price_discount = np.where(df_energy.DayOfWeek >= 5, -15, 0)

# Solar correlation: negative correlation with solar generation (more solar = lower prices)
solar_effect = -20 * (df_energy.HomeGeneration / 8)  # Normalize to max generation

# Add price volatility and occasional negative prices
price_volatility = np.random.normal(0, 25, len(df_energy))

# Occasional negative price events (especially during high solar periods)
negative_price_prob = 0.02 * (df_energy.HomeGeneration / 8)  # Higher prob during high solar
negative_price_events = np.random.random(len(df_energy)) < negative_price_prob
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

df_energy = df_energy.assign(PriceSignal = total_price)

# Save data
df_energy.to_parquet('cache/df_energy.parquet')

battery_setup = Battery(
    capacity               = 13.5,  # kWh (typical Tesla Powerwall size)
    max_charge_rate        = 5.0,   # kW
    max_discharge_rate     = 5.0,   # kW
    current_soc            = 6.75,  # kWh (50% of capacity)
    final_soc              = 6.75,  # kWh (same as current - maintain level)
    charging_efficiency    = 0.95,  # 95% efficiency when charging
    discharging_efficiency = 0.95,  # 95% efficiency when discharging
    soc_min                = 1.35,  # kWh (10% of capacity reserved)
    soc_max                = 13.5,  # kWh (same as capacity)
    grid_import_limit      = 17.25, # kW (3x25A typical home connection)
    grid_export_limit      = 17.25, # kW (same as import for simplicity)
)

# Save battery setup as a simple text representation for now
with open('cache/battery_setup.json', 'w') as f:
    f.write(battery_setup.model_dump_json(indent=4))

print(f"Generated {len(df_energy)} data points for 2024")
print(f"Date range: {df_energy.index[0]} to {df_energy.index[-1]}")
print(f"\nDataFrame columns: {list(df_energy.columns)}")
print(f"\nBattery setup: {battery_setup}")
print(f"\nSample data preview:")
print(df_energy.head())
print(f"\nPrice signal stats (€/MWh):")
print(f"Min: {df_energy.PriceSignal.min():.1f}")
print(f"Max: {df_energy.PriceSignal.max():.1f}")  
print(f"Mean: {df_energy.PriceSignal.mean():.1f}")
print(f"Negative prices: {(df_energy.PriceSignal < 0).sum()} events")
