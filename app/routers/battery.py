from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
import numpy as np
import pulp as pl

from ..schemas.battery import (
    BatteryOptimizationRequest,
    BatteryOptimizationResponse,
    OptimizationResult
)

router = APIRouter(prefix="/battery", tags=["battery"])


@router.post("/optimize", response_model=BatteryOptimizationResponse)
async def optimize_battery(request: BatteryOptimizationRequest) -> BatteryOptimizationResponse:
    """
    Optimize battery charging/discharging schedule for time-arbitrage profit maximization.
    
    Uses linear programming to find optimal battery control strategy given:
    - Energy generation/consumption forecasts
    - Battery technical specifications  
    - Grid connection limits
    - Electricity price forecasts
    
    All inputs must use UTC timezone and consistent kWh energy units.
    Maximum 2000 timesteps supported.
    """
    try:
        # Validate and parse input data
        df_energy = _validate_and_parse_energy_data(request)
        interval_hours = request.interval_seconds / 3600
        
        # Set up optimization problem
        optimization_results = _solve_battery_optimization(
            df_energy=df_energy,
            battery=request.battery,
            grid_limits=request.grid_limits,
            interval_hours=interval_hours
        )
        
        return optimization_results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


def _validate_and_parse_energy_data(request: BatteryOptimizationRequest) -> pd.DataFrame:
    """Convert request data to pandas DataFrame with validation."""
    # Convert to DataFrame
    energy_data = []
    for point in request.energy_data:
        energy_data.append({
            'timestamp': point.timestamp,
            'generation_kwh': point.generation_kwh,
            'consumption_kwh': point.consumption_kwh, 
            'price_eur_per_mwh': point.price_eur_per_mwh
        })
    
    df = pd.DataFrame(energy_data)
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Validate timestep intervals
    if len(df) > 2000:
        raise ValueError(f"Too many timesteps: {len(df)}. Maximum 2000 allowed.")
    
    if len(df) < 2:
        raise ValueError("At least 2 timesteps required for optimization.")
    
    # Check intervals are evenly spaced
    intervals = df.index[1:] - df.index[:-1]
    expected_interval = pd.Timedelta(seconds=request.interval_seconds)
    
    if not all(intervals == expected_interval):
        raise ValueError(
            f"Timesteps must be evenly spaced at {request.interval_seconds} second intervals. "
            f"Found irregular intervals."
        )
    
    return df


def _solve_battery_optimization(
    df_energy: pd.DataFrame,
    battery,
    grid_limits, 
    interval_hours: float
) -> BatteryOptimizationResponse:
    """Port of optimization logic from run.py lines 311-400."""
    
    n_steps = len(df_energy)
    timesteps = range(n_steps)
    
    # Create optimization problem
    prob = pl.LpProblem("Battery_Optimization", pl.LpMaximize)
    
    # Decision variables (all in kWh per interval)
    battery_charge = pl.LpVariable.dicts(
        "BatteryCharge",
        timesteps,
        lowBound=0,
        upBound=battery.max_charge_rate_kwh_per_interval
    )
    battery_discharge = pl.LpVariable.dicts(
        "BatteryDischarge", 
        timesteps,
        lowBound=0,
        upBound=battery.max_discharge_rate_kwh_per_interval
    )
    grid_import = pl.LpVariable.dicts(
        "GridImport",
        timesteps,
        lowBound=0,
        upBound=grid_limits.max_import_kwh_per_interval
    )
    grid_export = pl.LpVariable.dicts(
        "GridExport",
        timesteps, 
        lowBound=0,
        upBound=grid_limits.max_export_kwh_per_interval
    )
    soc = pl.LpVariable.dicts(
        "SoC",
        timesteps,
        lowBound=battery.min_soc_kwh,
        upBound=battery.max_soc_kwh
    )
    
    # Objective: maximize profit
    price_per_kwh = df_energy.price_eur_per_mwh / 1000
    profit = pl.lpSum([
        grid_export[t] * price_per_kwh.iloc[t] -  # Revenue from export
        grid_import[t] * price_per_kwh.iloc[t]    # Cost of import  
        for t in timesteps
    ])
    prob += profit
    
    # Constraints
    for t in timesteps:
        # Power balance constraint (all in kWh per interval)
        prob += (
            df_energy.generation_kwh.iloc[t] +
            battery_discharge[t] + grid_import[t]
            ==
            df_energy.consumption_kwh.iloc[t] +
            battery_charge[t] + grid_export[t]
        )
        
        # SoC dynamics with efficiency losses
        if t == 0:
            # Initial SoC
            prob += soc[t] == (
                battery.current_soc_kwh +
                battery_charge[t] * battery.charging_efficiency -
                battery_discharge[t] / battery.discharging_efficiency
            )
        else:
            # SoC evolution  
            prob += soc[t] == (
                soc[t-1] +
                battery_charge[t] * battery.charging_efficiency -
                battery_discharge[t] / battery.discharging_efficiency
            )
    
    # Final SoC constraint
    prob += soc[n_steps - 1] == battery.target_soc_kwh
    
    # Solve the problem
    prob.solve(pl.PULP_CBC_CMD(msg=0))
    
    # Check if solution is optimal
    if prob.status != pl.LpStatusOptimal:
        raise ValueError(f"Optimization failed with status: {pl.LpStatus[prob.status]}")
    
    # Extract results
    results = []
    total_profit = 0
    
    for t in timesteps:
        timestamp = df_energy.index[t]
        battery_charge_val = battery_charge[t].varValue or 0
        battery_discharge_val = battery_discharge[t].varValue or 0
        grid_import_val = grid_import[t].varValue or 0
        grid_export_val = grid_export[t].varValue or 0
        soc_val = soc[t].varValue or 0
        
        # Calculate profit for this timestep
        profit_eur = (
            grid_export_val * price_per_kwh.iloc[t] -
            grid_import_val * price_per_kwh.iloc[t]
        )
        total_profit += profit_eur
        
        results.append(OptimizationResult(
            timestamp=timestamp.isoformat(),
            battery_charge_kwh=battery_charge_val,
            battery_discharge_kwh=battery_discharge_val,
            grid_import_kwh=grid_import_val, 
            grid_export_kwh=grid_export_val,
            soc_kwh=soc_val,
            profit_eur=profit_eur
        ))
    
    return BatteryOptimizationResponse(
        status="optimal",
        results=results,
        total_profit_eur=total_profit
    )