"""SEWA (Sharjah Electricity and Water Authority) specific models"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from .base import UtilityBillBase, ConsumptionData, MeterReading


class SEWAMeter(BaseModel):
    """SEWA meter information"""
    meter_number: str
    meter_type: str  # "Electricity", "Water", "Gas"
    reading_current: Optional[str] = None
    reading_previous: Optional[str] = None
    consumption: Optional[float] = None
    multiplier: Optional[float] = 1.0


class SEWATariffSlab(BaseModel):
    """SEWA tariff slab information"""
    slab_range: str  # e.g., "1-200 Units"
    units: float
    rate: float
    amount: float


class SEWABill(UtilityBillBase):
    """SEWA utility bill model with specific fields"""
    # Provider defaults
    provider_name: str = "Sharjah Electricity and Water Authority (SEWA)"
    provider_region: str = "Sharjah, UAE"
    
    # SEWA-specific identifiers
    premise_id: str = Field(description="SEWA Premise ID")
    flat_number: Optional[str] = Field(None, description="Flat/apartment number")
    building_name: Optional[str] = None
    
    # Multiple meters support
    meters: Dict[str, SEWAMeter] = Field(default_factory=dict, description="All meters by type")
    
    # Consumption data
    electricity_consumption: Optional[ConsumptionData] = None
    water_consumption: Optional[ConsumptionData] = None
    gas_consumption: Optional[ConsumptionData] = None
    
    # Simplified fields
    electricity_kwh: Optional[float] = Field(None, description="Total electricity in kWh")
    water_m3: Optional[float] = Field(None, description="Total water in cubic meters")
    gas_consumption_value: Optional[float] = Field(None, description="Total gas consumption")
    gas_unit: Optional[str] = Field(None, description="Gas consumption unit")
    
    # Tariff breakdown
    electricity_slabs: Optional[List[SEWATariffSlab]] = None
    water_slabs: Optional[List[SEWATariffSlab]] = None
    gas_slabs: Optional[List[SEWATariffSlab]] = None
    
    # Service charges
    electricity_service_charge: Optional[float] = None
    water_service_charge: Optional[float] = None
    gas_service_charge: Optional[float] = None
    sewerage_charge: Optional[float] = None
    
    # Additional fees
    late_payment_fee: Optional[float] = None
    reconnection_fee: Optional[float] = None
    security_deposit: Optional[float] = None
    
    # Payment status
    payment_status: Optional[str] = None  # "Paid", "Unpaid", "Overdue"
    days_overdue: Optional[int] = None
    
    # Service status
    service_status: Optional[str] = None  # "Active", "Disconnected"
    disconnection_date: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "account_number": "7965198366",
                "premise_id": "PR123456",
                "flat_number": "101",
                "bill_date": "15/02/2020",
                "total_amount": 445.65,
                "electricity_kwh": 358.0,
                "water_m3": 121.3
            }
        }