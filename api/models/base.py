"""Base Pydantic models for utility bills"""

from typing import Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class MeterReading(BaseModel):
    """Meter reading information"""
    current: str
    previous: str
    consumption: Optional[float] = None
    reading_type: str = "actual"  # actual or estimated
    meter_number: Optional[str] = None


class ConsumptionData(BaseModel):
    """Consumption data for a utility service"""
    value: float
    unit: str
    data_quality: str = "measured"
    meter_reading: Optional[MeterReading] = None


class EmissionsData(BaseModel):
    """Carbon emissions data"""
    total_co2e: float = Field(description="Total CO2 equivalent in kg")
    unit: str = "kgCO2e"


class BillingPeriod(BaseModel):
    """Billing period information"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    periodicity: str = "Monthly"
    days: Optional[int] = None


class UtilityBillBase(BaseModel):
    """Base model for all utility bills"""
    # Provider information
    provider_name: str
    provider_region: Optional[str] = None
    
    # Account information
    account_number: str
    customer_name: Optional[str] = None
    service_address: Optional[str] = None
    premise_id: Optional[str] = None
    
    # Bill information
    bill_date: str
    due_date: Optional[str] = None
    billing_period: Optional[BillingPeriod] = None
    invoice_number: Optional[str] = None
    
    # Financial information
    total_amount: float
    currency: str = "AED"
    vat_amount: Optional[float] = None
    vat_rate: Optional[float] = 0.05
    
    # Extraction metadata
    extraction_confidence: Optional[float] = None
    extraction_timestamp: Optional[datetime] = None
    source_document: Optional[str] = None