"""Waste management bill model with recycling tracking"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from qwen_vl_integration.src.models.base import UtilityBillBase


class WasteStream(BaseModel):
    """Individual waste stream data"""
    stream_type: str = Field(description="Type of waste (e.g., general, recyclable, organic, hazardous)")
    quantity_kg: Optional[float] = Field(None, description="Quantity in kilograms")
    quantity_volume: Optional[float] = Field(None, description="Quantity in volume (m3 or liters)")
    volume_unit: Optional[str] = Field(None, description="Unit for volume measurement")
    collection_frequency: Optional[str] = Field(None, description="Collection frequency (e.g., weekly, bi-weekly)")
    diversion_rate: Optional[float] = Field(None, description="Percentage diverted from landfill")


class RecyclingData(BaseModel):
    """Recycling performance data"""
    total_recycled_kg: float = Field(description="Total recycled in kg")
    recycling_rate: float = Field(description="Recycling rate as percentage")
    materials_breakdown: Optional[Dict[str, float]] = Field(None, description="Breakdown by material type")
    contamination_rate: Optional[float] = Field(None, description="Contamination rate percentage")


class WasteEmissions(BaseModel):
    """Emissions from waste management"""
    total_co2e_kg: float = Field(description="Total CO2 equivalent from waste")
    landfill_emissions_kg: Optional[float] = Field(None, description="Emissions from landfilled waste")
    avoided_emissions_kg: Optional[float] = Field(None, description="Emissions avoided through recycling")
    methane_captured: Optional[bool] = Field(None, description="Whether methane is captured at landfill")


class WasteBill(UtilityBillBase):
    """Model for waste management bills"""
    # Waste quantities
    total_waste_kg: float = Field(description="Total waste generated in kg")
    waste_streams: List[WasteStream] = Field(description="Breakdown by waste stream")
    
    # Recycling
    recycling_data: RecyclingData = Field(description="Recycling performance data")
    
    # Service information
    collection_service: str = Field(description="Type of collection service")
    bin_sizes: Optional[Dict[str, str]] = Field(None, description="Bin sizes by waste type")
    collection_schedule: Optional[Dict[str, str]] = Field(None, description="Collection schedule by waste type")
    
    # Charges
    waste_collection_charge: Optional[float] = Field(None, description="Waste collection charge")
    recycling_credit: Optional[float] = Field(None, description="Credit for recycling")
    landfill_tax: Optional[float] = Field(None, description="Landfill tax if applicable")
    special_waste_charge: Optional[float] = Field(None, description="Charge for special waste items")
    
    # Environmental impact
    waste_emissions: Optional[WasteEmissions] = Field(None, description="Emissions data")
    landfill_diversion_rate: Optional[float] = Field(None, description="Overall landfill diversion rate")
    
    # Compliance
    waste_hierarchy_compliance: Optional[bool] = Field(None, description="Compliance with waste hierarchy")
    zero_waste_target: Optional[float] = Field(None, description="Zero waste target percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider_name": "Dubai Municipality",
                "account_number": "WM123456",
                "bill_date": "2024-01-15",
                "total_amount": 85.00,
                "total_waste_kg": 120.5,
                "waste_streams": [
                    {
                        "stream_type": "general",
                        "quantity_kg": 75.0,
                        "collection_frequency": "weekly"
                    },
                    {
                        "stream_type": "recyclable",
                        "quantity_kg": 35.5,
                        "collection_frequency": "bi-weekly",
                        "diversion_rate": 100.0
                    },
                    {
                        "stream_type": "organic",
                        "quantity_kg": 10.0,
                        "collection_frequency": "weekly",
                        "diversion_rate": 100.0
                    }
                ],
                "recycling_data": {
                    "total_recycled_kg": 45.5,
                    "recycling_rate": 37.8,
                    "materials_breakdown": {
                        "plastic": 15.0,
                        "paper": 20.5,
                        "glass": 5.0,
                        "metal": 5.0
                    }
                },
                "collection_service": "Residential",
                "waste_collection_charge": 85.00,
                "landfill_diversion_rate": 37.8
            }
        }