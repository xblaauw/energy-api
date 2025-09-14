from pydantic import BaseModel, field_validator, Field
from typing import List
import pandas as pd
from datetime import datetime


class EnergyDataPoint(BaseModel):
    """Single timestep of energy data with strict UTC timezone validation."""
    timestamp        : str   = Field(..., description="ISO format UTC timestamp (e.g., '2024-06-01T00:00:00Z')")
    generation_kwh   : float = Field(..., ge=0, description="Energy generation in kWh for this interval")
    consumption_kwh  : float = Field(..., ge=0, description="Energy consumption in kWh for this interval")
    price_eur_per_mwh: float = Field(..., description="Electricity price in EUR/MWh")
    
    @field_validator('timestamp')
    @classmethod
    def validate_utc_timestamp(cls, v: str) -> str:
        """Validate timestamp is UTC and ISO format."""
        try:
            # Parse with pandas for robust timezone handling
            ts = pd.Timestamp(v)
            
            # Must have timezone info
            if ts.tz is None:
                raise ValueError("Timestamp must include timezone information")
            
            # Must be UTC - check using string representation since pandas handles this differently
            if str(ts.tz) != 'UTC':
                raise ValueError("Timestamp must be in UTC timezone")
            
            # Return as ISO format string
            return ts.isoformat()
            
        except Exception as e:
            raise ValueError(f"Invalid timestamp format. Use ISO format with UTC timezone (e.g., '2024-06-01T00:00:00Z'): {e}")


class BatterySpecs(BaseModel):
    """Battery technical specifications - all energy values in kWh."""
    capacity_kwh                       : float = Field(..., gt=0, description="Total battery capacity in kWh")
    max_charge_rate_kwh_per_interval   : float = Field(..., gt=0, description="Max energy that can be charged per interval in kWh")
    max_discharge_rate_kwh_per_interval: float = Field(..., gt=0, description="Max energy that can be discharged per interval in kWh")
    charging_efficiency                : float = Field(..., gt=0, le=1, description="Charging efficiency (0-1)")
    discharging_efficiency             : float = Field(..., gt=0, le=1, description="Discharging efficiency (0-1)")
    current_soc_kwh                    : float = Field(..., ge=0, description="Current state of charge in kWh")
    target_soc_kwh                     : float = Field(..., ge=0, description="Target final state of charge in kWh")
    min_soc_kwh                        : float = Field(..., ge=0, description="Minimum allowed state of charge in kWh")
    max_soc_kwh                        : float = Field(..., gt=0, description="Maximum allowed state of charge in kWh")
    
    @field_validator('current_soc_kwh', 'target_soc_kwh')
    @classmethod  
    def validate_soc_within_bounds(cls, v, info):
        """Validate SoC values are within min/max bounds."""
        # Note: This runs before all fields are set, so we can't cross-validate here
        # Cross-validation will be done at the request level
        return v
    
    @field_validator('max_soc_kwh')
    @classmethod
    def validate_max_soc_le_capacity(cls, v, info):
        """Validate max SoC doesn't exceed capacity."""
        if 'capacity_kwh' in info.data and v > info.data['capacity_kwh']:
            raise ValueError("max_soc_kwh cannot exceed capacity_kwh")
        return v


class GridLimits(BaseModel):
    """Grid connection limits - all power values converted to energy per interval."""
    max_import_kwh_per_interval: float = Field(..., gt=0, description="Max energy import per interval in kWh")
    max_export_kwh_per_interval: float = Field(..., gt=0, description="Max energy export per interval in kWh")


class BatteryOptimizationRequest(BaseModel):
    """Complete battery optimization request."""
    energy_data     : List[EnergyDataPoint] = Field(..., max_length=2000, min_length=2, description="Energy forecast data (max 2000 timesteps)")
    interval_seconds: int = Field(..., gt=0, le=3600, description="Time interval between data points in seconds (max 1 hour)")
    battery         : BatterySpecs
    grid_limits     : GridLimits
    
    @field_validator('energy_data')
    @classmethod
    def validate_energy_data_length(cls, v):
        """Validate energy data constraints."""
        if len(v) < 2:
            raise ValueError("At least 2 timesteps required")
        if len(v) > 2000:
            raise ValueError("Maximum 2000 timesteps allowed")
        return v
    
    def model_post_init(self, __context) -> None:
        """Cross-validate battery SoC bounds after all fields are set."""
        battery = self.battery
        
        # Validate SoC bounds
        if battery.min_soc_kwh > battery.max_soc_kwh:
            raise ValueError("min_soc_kwh cannot exceed max_soc_kwh")
        
        if not (battery.min_soc_kwh <= battery.current_soc_kwh <= battery.max_soc_kwh):
            raise ValueError("current_soc_kwh must be within min_soc_kwh and max_soc_kwh bounds")
            
        if not (battery.min_soc_kwh <= battery.target_soc_kwh <= battery.max_soc_kwh):
            raise ValueError("target_soc_kwh must be within min_soc_kwh and max_soc_kwh bounds")


class OptimizationResult(BaseModel):
    """Single timestep optimization result."""
    timestamp            : str   = Field(..., description="UTC timestamp for this result")
    battery_charge_kwh   : float = Field(..., ge=0, description="Battery energy charged in kWh")
    battery_discharge_kwh: float = Field(..., ge=0, description="Battery energy discharged in kWh")
    grid_import_kwh      : float = Field(..., ge=0, description="Energy imported from grid in kWh")
    grid_export_kwh      : float = Field(..., ge=0, description="Energy exported to grid in kWh")
    soc_kwh              : float = Field(..., ge=0, description="Battery state of charge in kWh")
    profit_eur           : float = Field(..., description="Profit/loss for this timestep in EUR")


class BatteryOptimizationResponse(BaseModel):
    """Complete optimization response."""
    status          : str                      = Field(..., description="Optimization status (e.g., 'optimal', 'infeasible')")
    results         : List[OptimizationResult] = Field(..., description="Timestep-by-timestep optimization results")
    total_profit_eur: float                    = Field(..., description="Total profit over all timesteps in EUR")