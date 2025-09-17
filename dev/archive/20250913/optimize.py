import pandas as pd
import pulp as pl
from models import Battery

def optimize_battery(battery: Battery, df_energy: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """
    Optimize battery operation to maximize profit over given time series.
    Args:
        battery_setup: BatterySetup model with battery parameters
        df_energy: DataFrame with PriceSignal, HomeGeneration, HomeConsumption columns
        interval_minutes: time interval between rows in minutes
    Returns:
        DataFrame with added optimization result columns
    """
    # Convert interval to hours for energy calculations
    interval_hours = interval_minutes / 60.0
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
        upBound=battery.grid_import_limit
    )
    grid_export = pl.LpVariable.dicts(
        "GridExport",
        timesteps,
        lowBound=0,
        upBound=battery.grid_export_limit
    )
    soc = pl.LpVariable.dicts(
        "SoC",
        timesteps,
        lowBound=battery.soc_min,
        upBound=battery.soc_max
    )

    # Objective: maximize profit
    price_per_kwh = df_energy.PriceSignal / 1000
    profit = pl.lpSum([
        grid_export[t] * price_per_kwh.iloc[t] * interval_hours -  # Revenue from export
        grid_import[t] * price_per_kwh.iloc[t] * interval_hours     # Cost of import
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
    print(f"Total profit with battery:            €{battery_final:.2f}")
    print(f"Total profit without battery (baseline): €{baseline_final:.2f}")
    print(f"Difference:                           €{(battery_final - baseline_final):.2f}")
    print(f"Final SoC: {df_result.SoC.iloc[-1]:.2f} kWh (target: {battery.final_soc} kWh)")

    return df_result


# Test with sample data (uncomment to run)
if __name__ == "__main__":
    import sys; is_jupyter = 'ipykernel' in sys.modules

    if is_jupyter:
        t0 = pd.to_datetime('2024-01-01')
        t1 = pd.to_datetime('2024-01-08')
    
    else:
        import argparse

        # Setup argument parser with optional positional arguments and defaults
        parser = argparse.ArgumentParser(description="Run battery optimization on a time window.")
        parser.add_argument("start_date", nargs='?', type=str, default='2024-01-01', help="Start date (YYYY-MM-DD)")
        parser.add_argument("end_date", nargs='?', type=str, default='2024-01-08', help="End date (YYYY-MM-DD)")

        args = parser.parse_args()

        # Convert date string to pandas datetime
        t0 = pd.to_datetime(args.start_date)
        t1 = pd.to_datetime(args.end_date)

    # Load sample data
    battery_setup = Battery.model_validate_json(open('cache/battery_setup.json', 'r').read())
    df_energy = pd.read_parquet('cache/df_energy.parquet')

    # Filter time range
    df_energy = df_energy[t0:t1]  # Assumes df_energy.index is DatetimeIndex

    if len(df_energy) == 0:
        raise ValueError(f"No data found in range {t0.strftime('%d-%b-%Y')} to {t1.strftime('%d-%b-%Y')}")

    # Run optimization
    df_result = optimize_battery(battery_setup, df_energy, interval_minutes=15)

    # Save results
    df_result.to_parquet('cache/df_result.parquet')  
