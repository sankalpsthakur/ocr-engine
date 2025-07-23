"""Water bill model with consumption tracking"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from qwen_vl_integration.src.models.base import UtilityBillBase, ConsumptionData, MeterReading


class WaterQuality(BaseModel):
    """Water quality metrics"""
    parameter: str = Field(description="Quality parameter (e.g., pH, TDS, chlorine)")
    value: float = Field(description="Measured value")
    unit: str = Field(description="Unit of measurement")
    standard_limit: Optional[float] = Field(None, description="Regulatory standard limit")
    compliant: bool = Field(default=True, description="Whether value is within standards")


class WaterConsumption(ConsumptionData):
    """Water-specific consumption data"""
    consumption_m3: float = Field(description="Water consumption in cubic meters")
    consumption_gallons: Optional[float] = Field(None, description="Water consumption in gallons")
    daily_average_m3: Optional[float] = Field(None, description="Daily average consumption")
    tier: Optional[str] = Field(None, description="Consumption tier for pricing")
    

class WaterBill(UtilityBillBase):
    """Model for water utility bills"""
    # Water consumption
    water_consumption: WaterConsumption = Field(description="Water consumption data")
    water_m3: float = Field(description="Total water consumed in cubic meters")
    
    # Sewerage/Wastewater
    sewerage_m3: Optional[float] = Field(None, description="Sewerage volume (often same as water)")
    sewerage_charge: Optional[float] = Field(None, description="Sewerage service charge")
    
    # Service charges
    water_charge: Optional[float] = Field(None, description="Water usage charge")
    meter_service_charge: Optional[float] = Field(None, description="Meter service charge")
    infrastructure_charge: Optional[float] = Field(None, description="Infrastructure maintenance charge")
    
    # Conservation
    conservation_target_m3: Optional[float] = Field(None, description="Conservation target")
    conservation_achieved: Optional[bool] = Field(None, description="Whether conservation target met")
    water_saved_m3: Optional[float] = Field(None, description="Water saved compared to baseline")
    
    # Quality data
    water_quality_data: Optional[List[WaterQuality]] = Field(None, description="Water quality parameters")
    
    # Additional metrics
    water_intensity: Optional[float] = Field(None, description="Water intensity (m3/person or m3/sqm)")
    leak_detected: Optional[bool] = Field(None, description="Whether leak was detected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider_name": "DEWA",
                "account_number": "1234567890",
                "bill_date": "2024-01-15",
                "total_amount": 125.75,
                "water_m3": 25.5,
                "water_consumption": {
                    "value": 25.5,
                    "unit": "m3",
                    "consumption_m3": 25.5,
                    "daily_average_m3": 0.85,
                    "tier": "Tier 1"
                },
                "sewerage_m3": 25.5,
                "water_charge": 75.50,
                "sewerage_charge": 38.25,
                "conservation_achieved": True
            }
        }