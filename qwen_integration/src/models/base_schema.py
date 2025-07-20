"""Base Pydantic models for utility bill data extraction"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from enum import Enum


class DataQuality(str, Enum):
    """Data quality indicators"""
    MEASURED = "measured"
    ESTIMATED = "estimated"


class BillingPeriod(BaseModel):
    """Billing period information"""
    start_date: Optional[str] = Field(None, pattern=r'^\d{2}/\d{2}/\d{4}$|^$')
    end_date: Optional[str] = Field(None, pattern=r'^\d{2}/\d{2}/\d{4}$|^$')
    periodicity: str = Field(default="Monthly")


class MeterReading(BaseModel):
    """Meter reading information"""
    current: str = Field(default="")
    previous: str = Field(default="")
    reading_type: str = Field(default="actual")
    
    @field_validator('current', 'previous')
    @classmethod
    def validate_reading(cls, v: str) -> str:
        """Validate meter readings are numeric or empty"""
        if v and not v.isdigit():
            # Try to extract numeric part
            numeric_part = ''.join(filter(str.isdigit, v))
            if numeric_part:
                return numeric_part
            raise ValueError('Meter reading must be numeric')
        return v


class ConsumptionValue(BaseModel):
    """Consumption value with unit"""
    value: float = Field(default=0.0)
    unit: str
    
    @field_validator('value')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Ensure value is non-negative"""
        return max(0.0, v)


class ElectricityConsumption(BaseModel):
    """Electricity consumption data"""
    value: float = Field(default=0.0)
    unit: str = Field(default="kWh")
    data_quality: DataQuality = Field(default=DataQuality.MEASURED)
    meter_reading: Optional[MeterReading] = None


class WaterConsumption(BaseModel):
    """Water consumption data"""
    value: float = Field(default=0.0)
    unit: str = Field(default="m³")
    data_quality: DataQuality = Field(default=DataQuality.MEASURED)
    meter_reading: Optional[MeterReading] = None


class EmissionFactorReference(BaseModel):
    """Emission factor reference information"""
    region: str = Field(default="United Arab Emirates")
    grid_mix: str = Field(default="UAE_GRID_2024")
    year: str = Field(default="2024")


class EmissionsData(BaseModel):
    """Emissions data structure"""
    scope2: Dict[str, Any] = Field(default_factory=dict)


class ValidationInfo(BaseModel):
    """Validation information"""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_method: str = Field(default="qwen3_0.6b_surya")
    manual_verification_required: bool = Field(default=False)
    
    @field_validator('confidence')
    @classmethod
    def check_verification_threshold(cls, v: float, info) -> float:
        """Set manual verification flag based on confidence"""
        if v < 0.8:
            info.data['manual_verification_required'] = True
        return v


class Metadata(BaseModel):
    """Document metadata"""
    source_document: str
    page_numbers: List[int] = Field(default_factory=lambda: [1])
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sha256: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class BaseBillInfo(BaseModel):
    """Base bill information"""
    provider_name: str
    account_number: str = Field(default="")
    billing_period: BillingPeriod = Field(default_factory=BillingPeriod)
    bill_date: str = Field(default="")
    
    @field_validator('bill_date')
    @classmethod
    def validate_bill_date(cls, v: str) -> str:
        """Validate bill date format"""
        if v and not (len(v) == 10 and v[2] == '/' and v[5] == '/'):
            # Try to parse and reformat
            try:
                # Handle various date formats
                for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                    try:
                        date_obj = datetime.strptime(v, fmt)
                        return date_obj.strftime('%d/%m/%Y')
                    except ValueError:
                        continue
            except:
                pass
        return v


class BaseConsumptionData(BaseModel):
    """Base consumption data"""
    electricity: Optional[ElectricityConsumption] = None
    water: Optional[WaterConsumption] = None
    renewable_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    peak_demand: ConsumptionValue = Field(
        default_factory=lambda: ConsumptionValue(value=0.0, unit="kW")
    )


class BaseExtractedData(BaseModel):
    """Base extracted data structure"""
    bill_info: BaseBillInfo
    consumption_data: BaseConsumptionData
    emission_factor_reference: EmissionFactorReference = Field(default_factory=EmissionFactorReference)
    emissions_data: Optional[EmissionsData] = None


class BaseUtilityBill(BaseModel):
    """Base utility bill model"""
    document_type: str = Field(default="utility_bill")
    extracted_data: BaseExtractedData
    validation: ValidationInfo = Field(default_factory=ValidationInfo)
    metadata: Metadata