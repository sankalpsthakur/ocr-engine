"""Pydantic models for utility bill data extraction"""

from .converters import convert_existing_dewa_schema, convert_existing_sewa_schema
from .base_schema import (
    DataQuality,
    BillingPeriod,
    MeterReading,
    ConsumptionValue,
    ElectricityConsumption,
    WaterConsumption,
    EmissionFactorReference,
    EmissionsData,
    ValidationInfo,
    Metadata,
    BaseBillInfo,
    BaseConsumptionData,
    BaseExtractedData,
    BaseUtilityBill
)

from .dewa_schema import DEWABill, DEWABillInfo, DEWAConsumptionData, DEWAExtractedData
from .sewa_schema import SEWABill, SEWABillInfo, SEWAConsumptionData, SEWAExtractedData

__all__ = [
    # Converters
    'convert_existing_dewa_schema',
    'convert_existing_sewa_schema',
    # Base classes
    'DataQuality',
    'BillingPeriod',
    'MeterReading',
    'ConsumptionValue',
    'ElectricityConsumption',
    'WaterConsumption',
    'EmissionFactorReference',
    'EmissionsData',
    'ValidationInfo',
    'Metadata',
    'BaseBillInfo',
    'BaseConsumptionData',
    'BaseExtractedData',
    'BaseUtilityBill',
    # DEWA models
    'DEWABill',
    'DEWABillInfo',
    'DEWAConsumptionData',
    'DEWAExtractedData',
    # SEWA models
    'SEWABill',
    'SEWABillInfo',
    'SEWAConsumptionData',
    'SEWAExtractedData',
]