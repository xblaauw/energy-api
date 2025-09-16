from pydantic import BaseModel, Field, validator
from typing import Optional

class Battery(BaseModel):
    """Battery configuration and constraints for optimization."""
    
    # Battery physical parameters
    capacity: float = Field(..., gt=0, description="Battery capacity in kWh")
    max_charge_rate: float = Field(..., gt=0, description="Maximum charging power in kW")
    max_discharge_rate: float = Field(..., gt=0, description="Maximum discharging power in kW")
    
    # Battery state
    current_soc: float = Field(..., ge=0, description="Current state of charge in kWh")
    final_soc: float = Field(..., ge=0, description="Required final state of charge in kWh")
    
    # Battery efficiency
    charging_efficiency: float = Field(..., gt=0, le=1, description="Charging efficiency (0-1)")
    discharging_efficiency: float = Field(..., gt=0, le=1, description="Discharging efficiency (0-1)")
    
    # SoC limits
    soc_min: float = Field(..., ge=0, description="Minimum allowed SoC in kWh")
    soc_max: float = Field(..., gt=0, description="Maximum allowed SoC in kWh")
    
    @validator('current_soc', 'final_soc')
    def validate_soc_bounds(cls, v, values):
        """Ensure current and final SoC are within capacity bounds."""
        if 'capacity' in values and v > values['capacity']:
            raise ValueError('SoC cannot exceed battery capacity')
        return v
    
    @validator('soc_max')
    def validate_soc_max(cls, v, values):
        """Ensure soc_max doesn't exceed capacity."""
        if 'capacity' in values and v > values['capacity']:
            raise ValueError('soc_max cannot exceed battery capacity')
        return v
    
    @validator('soc_min')
    def validate_soc_min(cls, v, values):
        """Ensure soc_min is less than soc_max."""
        if 'soc_max' in values and v >= values['soc_max']:
            raise ValueError('soc_min must be less than soc_max')
        return v
    
    @validator('current_soc', 'final_soc')
    def validate_soc_within_limits(cls, v, values):
        """Ensure current and final SoC are within min/max limits."""
        if 'soc_min' in values and v < values['soc_min']:
            raise ValueError('SoC cannot be below soc_min')
        if 'soc_max' in values and v > values['soc_max']:
            raise ValueError('SoC cannot exceed soc_max')
        return v