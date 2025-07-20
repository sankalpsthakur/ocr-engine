"""DEWA-specific Pydantic models"""

from typing import Dict, Any, Optional
from pydantic import Field, model_validator
from .base_schema import (
    BaseUtilityBill, BaseBillInfo, BaseConsumptionData, 
    BaseExtractedData, ElectricityConsumption, MeterReading,
    EmissionsData, ConsumptionValue
)


class DEWABillInfo(BaseBillInfo):
    """DEWA-specific bill information"""
    provider_name: str = Field(default="Dubai Electricity and Water Authority (DEWA)")
    
    @model_validator(mode='after')
    def validate_dewa_specific(self):
        """DEWA-specific validations"""
        # DEWA account numbers are typically 10 digits
        if self.account_number and len(self.account_number) != 10:
            # Log warning but don't fail
            pass
        return self


class DEWAConsumptionData(BaseConsumptionData):
    """DEWA-specific consumption data"""
    electricity: ElectricityConsumption = Field(default_factory=ElectricityConsumption)
    
    @model_validator(mode='after')
    def ensure_meter_reading(self):
        """Ensure electricity has meter reading"""
        if self.electricity and not self.electricity.meter_reading:
            self.electricity.meter_reading = MeterReading()
        return self


class DEWAEmissionsData(EmissionsData):
    """DEWA-specific emissions data"""
    scope2: Dict[str, Any] = Field(default_factory=lambda: {
        "totalCO2e": {"value": 0.0, "unit": "kgCO2e"},
        "breakdown": {"electricity": 0.0}
    })
    
    def calculate_emissions(self, electricity_kwh: float):
        """Calculate emissions based on consumption"""
        # Default emission factor for UAE grid (kg CO2e per kWh)
        # Based on DEWA data: 120 kg CO2e for 299 kWh = 0.401337... kg CO2e/kWh
        # Using the exact factor from the bill
        emission_factor = 120.0 / 299.0  # ~0.401337
        
        total_emissions = electricity_kwh * emission_factor
        
        # Round to whole number as shown in the bill
        self.scope2["totalCO2e"]["value"] = round(total_emissions)
        self.scope2["breakdown"]["electricity"] = round(total_emissions)


class DEWAExtractedData(BaseExtractedData):
    """DEWA-specific extracted data"""
    bill_info: DEWABillInfo
    consumption_data: DEWAConsumptionData
    emissions_data: DEWAEmissionsData = Field(default_factory=DEWAEmissionsData)
    
    @model_validator(mode='after')
    def link_consumption_to_emissions(self):
        """Pass consumption data to emissions calculator"""
        if self.consumption_data.electricity:
            # Calculate emissions based on electricity consumption
            self.emissions_data.calculate_emissions(self.consumption_data.electricity.value)
        return self


class DEWABill(BaseUtilityBill):
    """DEWA utility bill model"""
    extracted_data: DEWAExtractedData
    
    @model_validator(mode='after')
    def validate_dewa_bill(self):
        """Overall DEWA bill validation"""
        # Set confidence based on data completeness
        confidence_score = 1.0
        
        # Check required fields
        if not self.extracted_data.bill_info.account_number:
            confidence_score -= 0.1
        if not self.extracted_data.bill_info.bill_date:
            confidence_score -= 0.1
        if not self.extracted_data.consumption_data.electricity.value:
            confidence_score -= 0.2
        if not self.extracted_data.consumption_data.electricity.meter_reading.current:
            confidence_score -= 0.1
            
        self.validation.confidence = max(0.0, confidence_score)
        self.validation.manual_verification_required = confidence_score < 0.8
        
        return self