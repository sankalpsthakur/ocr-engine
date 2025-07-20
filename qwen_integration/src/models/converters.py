"""Converters for existing schema formats to Pydantic models"""

from typing import Dict, Any


def convert_existing_dewa_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert existing DEWA schema format to Pydantic model format"""
    # Map camelCase to snake_case
    extracted = data.get("extractedData", {})
    bill_info = extracted.get("billInfo", {})
    consumption = extracted.get("consumptionData", {})
    
    converted = {
        "document_type": data.get("documentType", "utility_bill"),
        "extracted_data": {
            "bill_info": {
                "provider_name": bill_info.get("providerName", ""),
                "account_number": bill_info.get("accountNumber", ""),
                "bill_date": bill_info.get("billDate", ""),
                "billing_period": bill_info.get("billingPeriod", {})
            },
            "consumption_data": {
                "electricity": consumption.get("electricity", {}),
                "water": consumption.get("water", {}),
                "renewable_percentage": consumption.get("renewablePercentage", 0.0),
                "peak_demand": consumption.get("peakDemand", {"value": 0.0, "unit": "kW"})
            },
            "emission_factor_reference": extracted.get("emissionFactorReference", {}),
            "emissions_data": extracted.get("emissionsData", {})
        },
        "validation": data.get("validation", {}),
        "metadata": {
            "source_document": data.get("metadata", {}).get("sourceDocument", ""),
            "page_numbers": data.get("metadata", {}).get("pageNumbers", [1]),
            "extraction_timestamp": data.get("metadata", {}).get("extractionTimestamp", ""),
            "sha256": data.get("metadata", {}).get("sha256"),
            "processing_time_seconds": data.get("metadata", {}).get("processingTimeSeconds")
        }
    }
    
    # Handle meter reading conversion
    electricity = converted["extracted_data"]["consumption_data"]["electricity"]
    if electricity and "meterReading" in electricity:
        meter_reading = electricity["meterReading"]
        electricity["meter_reading"] = {
            "current": meter_reading.get("current", ""),
            "previous": meter_reading.get("previous", ""),
            "reading_type": meter_reading.get("readingType", "actual")
        }
        del electricity["meterReading"]
    
    # Handle data quality
    if electricity and "dataQuality" in electricity:
        electricity["data_quality"] = electricity.pop("dataQuality")
    
    return converted


def convert_existing_sewa_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert existing SEWA schema format to Pydantic model format"""
    # Similar to DEWA but handles water consumption
    converted = convert_existing_dewa_schema(data)
    
    # SEWA specific handling
    water = converted["extracted_data"]["consumption_data"].get("water", {})
    if water and "dataQuality" in water:
        water["data_quality"] = water.pop("dataQuality")
    
    # SEWA bills don't have emissions data
    if "emissions_data" in converted["extracted_data"]:
        converted["extracted_data"]["emissions_data"] = None
    
    return converted