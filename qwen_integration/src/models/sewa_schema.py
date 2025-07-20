"""SEWA-specific Pydantic models"""

from typing import Optional
from pydantic import Field, model_validator
from .base_schema import (
    BaseUtilityBill, BaseBillInfo, BaseConsumptionData, 
    BaseExtractedData, ElectricityConsumption, WaterConsumption,
    MeterReading
)


class SEWABillInfo(BaseBillInfo):
    """SEWA-specific bill information"""
    provider_name: str = Field(default="Sharjah Electricity and Water Authority (SEWA)")
    
    @model_validator(mode='after')
    def validate_sewa_specific(self):
        """SEWA-specific validations"""
        # SEWA account numbers are typically 10 digits
        if self.account_number and len(self.account_number) != 10:
            # Log warning but don't fail
            pass
        return self


class SEWAConsumptionData(BaseConsumptionData):
    """SEWA-specific consumption data - includes both electricity and water"""
    electricity: ElectricityConsumption = Field(default_factory=ElectricityConsumption)
    water: WaterConsumption = Field(default_factory=WaterConsumption)
    
    @model_validator(mode='after')
    def ensure_meter_readings(self):
        """Ensure both electricity and water have meter readings"""
        if self.electricity and not self.electricity.meter_reading:
            self.electricity.meter_reading = MeterReading()
        if self.water and not self.water.meter_reading:
            self.water.meter_reading = MeterReading()
        return self
    
    @model_validator(mode='after')
    def validate_water_unit(self):
        """Ensure water is in cubic meters"""
        if self.water and self.water.unit not in ["m³", "m3", "cubic meters"]:
            # Convert common units
            if self.water.unit.lower() in ["gallons", "gal"]:
                # Convert gallons to m³ (1 gallon = 0.00378541 m³)
                self.water.value = self.water.value * 0.00378541
                self.water.unit = "m³"
            elif self.water.unit.lower() in ["liters", "l"]:
                # Convert liters to m³ (1000 liters = 1 m³)
                self.water.value = self.water.value / 1000
                self.water.unit = "m³"
        return self


class SEWAExtractedData(BaseExtractedData):
    """SEWA-specific extracted data"""
    bill_info: SEWABillInfo
    consumption_data: SEWAConsumptionData
    # SEWA bills typically don't include emissions data
    emissions_data: Optional[None] = None
    
    @model_validator(mode='after')
    def validate_sewa_data(self):
        """SEWA-specific data validation"""
        # SEWA bills should have both electricity and water data
        if not self.consumption_data.electricity.value and not self.consumption_data.water.value:
            # Flag for manual review if both are missing
            pass
        return self


class SEWABill(BaseUtilityBill):
    """SEWA utility bill model"""
    extracted_data: SEWAExtractedData
    
    @model_validator(mode='after')
    def validate_sewa_bill(self):
        """Overall SEWA bill validation"""
        # Set confidence based on data completeness
        confidence_score = 1.0
        
        # Check required fields
        if not self.extracted_data.bill_info.account_number:
            confidence_score -= 0.1
        if not self.extracted_data.bill_info.bill_date:
            confidence_score -= 0.1
        if not self.extracted_data.consumption_data.electricity.value:
            confidence_score -= 0.15
        if not self.extracted_data.consumption_data.water.value:
            confidence_score -= 0.15
            
        # SEWA specific - both utilities should be present
        if (self.extracted_data.consumption_data.electricity.value == 0 or 
            self.extracted_data.consumption_data.water.value == 0):
            confidence_score -= 0.1
            
        self.validation.confidence = max(0.0, confidence_score)
        self.validation.manual_verification_required = confidence_score < 0.7
        
        return self