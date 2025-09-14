# %% Setup

# Core
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Installed
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

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

# Price forecast params
lookback_days          = 21
forecast_horizon_hours = 24


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


# %% Price Forecast Generation

# Not the focus of this project, so we will settle for simple and likely good enough
# Generate price forecast using rolling 3-week historical data
# This simulates daily forecasting where we predict next 24h based on past 3 weeks

# Compute steps
forecast_steps = forecast_horizon_hours * 4                  # 4 intervals per hour
lookback_steps = lookback_days * forecast_horizon_hours * 4  # lookback period in timesteps


def create_forecast_features(data):
    """Create features for price forecasting"""
    return np.column_stack([
        data.HourOfDay,
        np.sin(2 * np.pi * data.HourOfDay / 24),
        np.cos(2 * np.pi * data.HourOfDay / 24),
        data.DayOfWeek,
        np.sin(2 * np.pi * data.DayOfWeek / 7),
        np.cos(2 * np.pi * data.DayOfWeek / 7),
        data.DayOfYear,
        np.sin(2 * np.pi * data.DayOfYear / 365),
        np.cos(2 * np.pi * data.DayOfYear / 365),
        data.HomeGeneration,
        data.HomeConsumption,
    ])

forecast_prices = np.full(len(df_energy), np.nan)
start_idx = lookback_steps

print("Generating price forecast using 3-week rolling historical data...")

for i in range(start_idx, len(df_energy), forecast_steps):
    # Define the range for this forecast (next 24 hours or remaining data)
    forecast_end = min(i + forecast_steps, len(df_energy))

    # Get historical data for training (past 3 weeks)
    hist_start = max(0, i - lookback_steps)
    hist_data = df_energy.iloc[hist_start:i].copy()

    if len(hist_data) < 48:  # Need at least 2 days of data
        continue

    # Create features for historical data and forecast period
    hist_features = create_forecast_features(hist_data)
    hist_prices = hist_data['PriceSignal'].values

    forecast_data = df_energy.iloc[i:forecast_end].copy()
    forecast_features = create_forecast_features(forecast_data)

    # Train model and predict
    model = LinearRegression()
    model.fit(hist_features, hist_prices)
    predicted_prices = model.predict(forecast_features)

    # Apply smoothing (forecast can't capture all volatility)
    hist_std = np.std(hist_prices[-96:])  # Last 24 hours std
    pred_std = np.std(predicted_prices)

    if pred_std > 0:
        # Scale down volatility to 70% of historical
        smoothing_factor = 0.7 * hist_std / pred_std
        pred_mean = np.mean(predicted_prices)
        predicted_prices = pred_mean + (predicted_prices - pred_mean) * smoothing_factor

    # Apply rolling average for additional smoothing
    pred_df = pd.Series(predicted_prices)
    predicted_prices = pred_df.rolling(window=4, center=True, min_periods=1).mean().values

    # Store predictions
    forecast_prices[i:forecast_end] = predicted_prices

# Fill any remaining NaN values at the beginning with actual prices
mask = np.isnan(forecast_prices)
forecast_prices[mask] = df_energy['PriceSignal'].values[mask]

# Add forecast to dataframe
df_energy = df_energy.assign(PriceForecast=forecast_prices)

# Calculate forecast accuracy metrics
forecast_error = df_energy.PriceSignal - df_energy.PriceForecast
mae = np.mean(np.abs(forecast_error))
rmse = np.sqrt(np.mean(forecast_error**2))
mape = np.mean(np.abs(forecast_error / df_energy.PriceSignal)) * 100

print()
print(f"Price forecast generated successfully!")
print(f"Forecast accuracy metrics:")
print(f"Mean Absolute Error (MAE):     {mae:.2f} €/MWh")
print(f"Root Mean Square Error (RMSE): {rmse:.2f} €/MWh")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print()
print(f"Actual price range:   {df_energy.PriceSignal.min():.1f} to {df_energy.PriceSignal.max():.1f} €/MWh")
print(f"Forecast price range: {df_energy.PriceForecast.min():.1f} to {df_energy.PriceForecast.max():.1f} €/MWh")

# Show comparison
print()
print(f"Forecast vs Actual correlation: {df_energy.PriceSignal.corr(df_energy.PriceForecast):.3f}")


# %% Optimization via API

# Limit input data to specified time-range
df_energy = df_energy.loc[t0 : t1]

# Convert interval to hours for energy calculations
interval_hours = .25
interval_seconds = 900  # 15 minutes = 900 seconds

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE_URL}/battery/optimize"

print(f"Preparing API optimization call...")
print(f"Timesteps: {len(df_energy)}")

# Convert data for API (power to energy conversion)
energy_data = []
for idx, row in df_energy.iterrows():
    energy_data.append({
        "timestamp": idx.strftime('%Y-%m-%dT%H:%M:%SZ'),  # UTC ISO format
        "generation_kwh": (row.HomeGeneration * interval_hours).item(),    # Convert numpy to Python float
        "consumption_kwh": (row.HomeConsumption * interval_hours).item(),  # Convert numpy to Python float
        "price_eur_per_mwh": row.PriceForecast.item()  # Use forecast prices for optimization
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

print(f"Calling API at {API_ENDPOINT}...")

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
        print(f"API call successful!")
        print(f"Status: {api_result['status']}")
        print(f"Total profit: €{api_result['total_profit_eur']:.2f}")
    else:
        print(f"API call failed!")
        print(f"Response: {response.text}")
        raise ValueError("API optimization failed")

except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
    print("Make sure FastAPI server is running with: fastapi dev app/main.py")
    raise

# Process API results
api_results = []
for result in api_result['results']:
    api_results.append({
        'timestamp': pd.Timestamp(result['timestamp']).tz_convert(None),  # Remove timezone for local processing
        'BatteryCharge': result['battery_charge_kwh'] / interval_hours,        # Convert back to kW
        'BatteryDischarge': result['battery_discharge_kwh'] / interval_hours,  # Convert back to kW
        'GridImport': result['grid_import_kwh'] / interval_hours,              # Convert back to kW
        'GridExport': result['grid_export_kwh'] / interval_hours,              # Convert back to kW
        'SoC': result['soc_kwh'],                                              # Keep in kWh
    })

df_api_result = pd.DataFrame(api_results)
df_api_result.set_index('timestamp', inplace=True)

# Merge with original energy data
df_result = df_energy.join(df_api_result)

# Check if solution is optimal
if api_result['status'] != 'optimal':
    raise ValueError(f"Optimization failed with status: {api_result['status']}")

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
        "Price Signal vs Forecast",  # Updated title
        "Price Forecast vs Actual",   # New subplot
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

# Price Signal vs Forecast (both on same subplot for comparison)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.PriceSignal, name='Actual Price', line=dict(color='blue')), row=5, col=1)
fig.add_trace(go.Scatter(x=df_result.index, y=df_result.PriceForecast, name='Forecasted Price', line=dict(color='red', dash='dash')), row=5, col=1)

# Price Forecast vs Actual (difference/error)
price_error = df_result.PriceSignal - df_result.PriceForecast
fig.add_trace(go.Scatter(x=df_result.index, y=price_error, name='Forecast Error', line=dict(color='orange')), row=6, col=1)

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
    title="Battery Optimization Physics Check with Price Forecasting (API Version)",
    hovermode='x unified'
)

fig.update_xaxes(matches='x')

fig.show()