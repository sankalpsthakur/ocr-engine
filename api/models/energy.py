"""Energy bill model with carbon emissions tracking"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from .base import UtilityBillBase, ConsumptionData, EmissionsData


class EnergySourceBreakdown(BaseModel):
    """Breakdown of energy sources"""
    source_type: str = Field(description="Type of energy source (e.g., solar, natural_gas, grid)")
    percentage: float = Field(description="Percentage of total energy from this source")
    kwh: Optional[float] = Field(None, description="kWh from this source")


class CarbonEmissions(BaseModel):
    """Carbon emissions data for energy consumption"""
    total_co2e_kg: float = Field(description="Total CO2 equivalent in kg")
    emission_factor: float = Field(description="Emission factor used (kgCO2e/kWh)")
    calculation_method: str = Field(default="location_based", description="Method used for calculation")
    scope: str = Field(default="scope2", description="GHG Protocol scope (scope1, scope2, scope3)")
    renewable_percentage: Optional[float] = Field(None, description="Percentage of renewable energy")
    avoided_emissions_kg: Optional[float] = Field(None, description="Emissions avoided through renewable sources")


class EnergyConsumption(ConsumptionData):
    """Energy-specific consumption data"""
    peak_demand_kw: Optional[float] = Field(None, description="Peak demand in kW")
    power_factor: Optional[float] = Field(None, description="Power factor")
    energy_sources: Optional[List[EnergySourceBreakdown]] = Field(None, description="Breakdown by energy source")


class EnergyBill(UtilityBillBase):
    """Model for energy/electricity bills with carbon tracking"""
    # Energy consumption
    electricity_consumption: EnergyConsumption = Field(description="Electricity consumption data")
    electricity_kwh: float = Field(description="Total electricity consumed in kWh")
    
    # Carbon emissions
    carbon_emissions: CarbonEmissions = Field(description="Carbon emissions data")
    
    # Tariff information
    electricity_charge: Optional[float] = Field(None, description="Electricity usage charge")
    demand_charge: Optional[float] = Field(None, description="Demand charge if applicable")
    fuel_surcharge: Optional[float] = Field(None, description="Fuel adjustment surcharge")
    tariff_type: Optional[str] = Field(None, description="Tariff type (e.g., residential, commercial)")
    
    # Green energy
    renewable_energy_kwh: Optional[float] = Field(None, description="Renewable energy consumed")
    renewable_energy_credits: Optional[int] = Field(None, description="RECs purchased")
    
    # Additional metrics
    energy_intensity: Optional[float] = Field(None, description="Energy intensity (kWh/sqm)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider_name": "DEWA",
                "account_number": "1234567890",
                "bill_date": "2024-01-15",
                "total_amount": 450.50,
                "electricity_kwh": 1250.0,
                "electricity_consumption": {
                    "value": 1250.0,
                    "unit": "kWh",
                    "data_quality": "measured",
                    "peak_demand_kw": 5.2
                },
                "carbon_emissions": {
                    "total_co2e_kg": 625.0,
                    "emission_factor": 0.5,
                    "calculation_method": "location_based",
                    "scope": "scope2",
                    "renewable_percentage": 10.0
                }
            }
        }