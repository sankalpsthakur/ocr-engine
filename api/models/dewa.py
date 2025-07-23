"""DEWA (Dubai Electricity and Water Authority) specific models"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from .base import UtilityBillBase, ConsumptionData, EmissionsData, MeterReading


class DEWAMeterReading(MeterReading):
    """DEWA-specific meter reading with additional fields"""
    meter_type: str  # "Electricity" or "Water"
    meter_number: str
    location: Optional[str] = None


class DEWAConsumptionBreakdown(BaseModel):
    """Detailed consumption breakdown for DEWA bills"""
    slab: str  # e.g., "0-2000", "2001-4000"
    units: float
    rate_per_unit: float
    amount: float


class DEWABill(UtilityBillBase):
    """DEWA utility bill model with specific fields"""
    # Provider defaults
    provider_name: str = "Dubai Electricity and Water Authority (DEWA)"
    provider_region: str = "Dubai, UAE"
    
    # DEWA-specific identifiers
    vat_number: Optional[str] = Field(None, description="DEWA VAT registration number")
    contract_account: Optional[str] = Field(None, description="Contract account number")
    
    # Consumption data
    electricity_consumption: Optional[ConsumptionData] = None
    water_consumption: Optional[ConsumptionData] = None
    
    # Simplified fields for common access
    electricity_kwh: Optional[float] = Field(None, description="Total electricity consumption in kWh")
    water_m3: Optional[float] = Field(None, description="Total water consumption in cubic meters")
    
    # Meter readings
    meter_readings: Dict[str, DEWAMeterReading] = Field(default_factory=dict)
    
    # Consumption breakdown
    electricity_breakdown: Optional[List[DEWAConsumptionBreakdown]] = None
    water_breakdown: Optional[List[DEWAConsumptionBreakdown]] = None
    
    # Carbon emissions
    carbon_emissions: Optional[EmissionsData] = None
    carbon_kg_co2e: Optional[float] = Field(None, description="Total carbon emissions in kg CO2e")
    
    # Additional charges
    housing_fee: Optional[float] = None
    municipality_fee: Optional[float] = None
    fuel_surcharge: Optional[float] = None
    
    # Payment information
    last_payment_amount: Optional[float] = None
    last_payment_date: Optional[str] = None
    balance_forward: Optional[float] = None
    
    # Green initiatives
    renewable_percentage: Optional[float] = Field(None, description="Percentage of renewable energy")
    green_tariff_enrolled: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "account_number": "2052672303",
                "bill_date": "21/05/2025",
                "total_amount": 287.00,
                "electricity_kwh": 299.0,
                "carbon_kg_co2e": 120.0,
                "vat_number": "100330017500003"
            }
        }