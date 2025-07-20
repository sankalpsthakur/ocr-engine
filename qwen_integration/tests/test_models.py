"""Unit tests for Pydantic models"""

import pytest
from datetime import datetime
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_integration.src.models import (
    DEWABill, SEWABill,
    BillingPeriod, MeterReading,
    ElectricityConsumption, WaterConsumption,
    DataQuality,
    convert_existing_dewa_schema, convert_existing_sewa_schema
)


class TestBillingPeriod:
    """Test BillingPeriod model"""
    
    def test_valid_dates(self):
        """Test valid date formats"""
        period = BillingPeriod(
            start_date="01/01/2024",
            end_date="31/01/2024",
            periodicity="Monthly"
        )
        assert period.start_date == "01/01/2024"
        assert period.end_date == "31/01/2024"
        
    def test_empty_dates(self):
        """Test empty dates are allowed"""
        period = BillingPeriod(
            start_date="",
            end_date="",
            periodicity="Monthly"
        )
        assert period.start_date == ""
        assert period.end_date == ""
        
    def test_invalid_date_format(self):
        """Test invalid date format raises error"""
        with pytest.raises(ValueError):
            BillingPeriod(
                start_date="2024-01-01",  # Wrong format
                end_date="31/01/2024"
            )


class TestMeterReading:
    """Test MeterReading model"""
    
    def test_valid_readings(self):
        """Test valid numeric readings"""
        reading = MeterReading(
            current="12345",
            previous="12000",
            reading_type="actual"
        )
        assert reading.current == "12345"
        assert reading.previous == "12000"
        
    def test_numeric_extraction(self):
        """Test extraction of numeric part from mixed string"""
        reading = MeterReading(
            current="Reading: 12345",
            previous="12000 kWh"
        )
        assert reading.current == "12345"
        assert reading.previous == "12000"
        
    def test_empty_readings(self):
        """Test empty readings are allowed"""
        reading = MeterReading(current="", previous="")
        assert reading.current == ""
        assert reading.previous == ""


class TestDEWABill:
    """Test DEWA bill model"""
    
    def test_minimal_dewa_bill(self):
        """Test creating DEWA bill with minimal data"""
        bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "provider_name": "Dubai Electricity and Water Authority (DEWA)",
                    "account_number": "1234567890",
                    "bill_date": "15/01/2024"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 500.0,
                        "unit": "kWh"
                    }
                }
            },
            "metadata": {
                "source_document": "DEWA_test.png"
            }
        }
        
        bill = DEWABill(**bill_data)
        assert bill.extracted_data.bill_info.provider_name == "Dubai Electricity and Water Authority (DEWA)"
        assert bill.extracted_data.consumption_data.electricity.value == 500.0
        
    def test_dewa_with_meter_readings(self):
        """Test DEWA bill with meter readings"""
        bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "account_number": "1234567890",
                    "bill_date": "15/01/2024"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 300.0,
                        "unit": "kWh",
                        "meter_reading": {
                            "current": "19462",
                            "previous": "19162"
                        }
                    }
                }
            },
            "metadata": {
                "source_document": "DEWA_test.png"
            }
        }
        
        bill = DEWABill(**bill_data)
        assert bill.extracted_data.consumption_data.electricity.meter_reading.current == "19462"
        assert bill.extracted_data.consumption_data.electricity.meter_reading.previous == "19162"
        
    def test_dewa_confidence_calculation(self):
        """Test confidence score calculation"""
        # Complete bill data
        complete_bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "account_number": "1234567890",
                    "bill_date": "15/01/2024"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 300.0,
                        "unit": "kWh",
                        "meter_reading": {
                            "current": "19462",
                            "previous": "19162"
                        }
                    }
                }
            },
            "metadata": {
                "source_document": "DEWA_test.png"
            }
        }
        
        bill = DEWABill(**complete_bill_data)
        assert bill.validation.confidence >= 0.8
        assert not bill.validation.manual_verification_required
        
        # Incomplete bill data
        incomplete_bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "account_number": "",
                    "bill_date": ""
                },
                "consumption_data": {
                    "electricity": {
                        "value": 0.0,
                        "unit": "kWh"
                    }
                }
            },
            "metadata": {
                "source_document": "DEWA_test.png"
            }
        }
        
        bill = DEWABill(**incomplete_bill_data)
        assert bill.validation.confidence < 0.8
        assert bill.validation.manual_verification_required
        
    def test_dewa_emissions_calculation(self):
        """Test emissions are calculated based on consumption"""
        bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "account_number": "1234567890"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 100.0,  # 100 kWh
                        "unit": "kWh"
                    }
                }
            },
            "metadata": {
                "source_document": "DEWA_test.png"
            }
        }
        
        bill = DEWABill(**bill_data)
        # With default emission factor of 0.4 kg CO2e/kWh
        assert bill.extracted_data.emissions_data.scope2["totalCO2e"]["value"] == 40.0
        assert bill.extracted_data.emissions_data.scope2["breakdown"]["electricity"] == 40.0


class TestSEWABill:
    """Test SEWA bill model"""
    
    def test_minimal_sewa_bill(self):
        """Test creating SEWA bill with minimal data"""
        bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "provider_name": "Sharjah Electricity and Water Authority (SEWA)",
                    "account_number": "9876543210",
                    "bill_date": "20/02/2024"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 358.0,
                        "unit": "kWh"
                    },
                    "water": {
                        "value": 121.3,
                        "unit": "m³"
                    }
                }
            },
            "metadata": {
                "source_document": "SEWA_test.png"
            }
        }
        
        bill = SEWABill(**bill_data)
        assert bill.extracted_data.bill_info.provider_name == "Sharjah Electricity and Water Authority (SEWA)"
        assert bill.extracted_data.consumption_data.electricity.value == 358.0
        assert bill.extracted_data.consumption_data.water.value == 121.3
        
    def test_sewa_water_unit_conversion(self):
        """Test water unit conversions"""
        bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "account_number": "9876543210"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 100.0,
                        "unit": "kWh"
                    },
                    "water": {
                        "value": 1000.0,
                        "unit": "liters"
                    }
                }
            },
            "metadata": {
                "source_document": "SEWA_test.png"
            }
        }
        
        bill = SEWABill(**bill_data)
        # 1000 liters = 1 m³
        assert bill.extracted_data.consumption_data.water.value == 1.0
        assert bill.extracted_data.consumption_data.water.unit == "m³"
        
    def test_sewa_confidence_with_both_utilities(self):
        """Test confidence calculation for SEWA with both utilities"""
        bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "account_number": "9876543210",
                    "bill_date": "20/02/2024"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 358.0,
                        "unit": "kWh"
                    },
                    "water": {
                        "value": 121.3,
                        "unit": "m³"
                    }
                }
            },
            "metadata": {
                "source_document": "SEWA_test.png"
            }
        }
        
        bill = SEWABill(**bill_data)
        assert bill.validation.confidence >= 0.7
        assert not bill.validation.manual_verification_required
        
    def test_sewa_missing_water_reduces_confidence(self):
        """Test that missing water data reduces confidence"""
        bill_data = {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "account_number": "9876543210",
                    "bill_date": "20/02/2024"
                },
                "consumption_data": {
                    "electricity": {
                        "value": 358.0,
                        "unit": "kWh"
                    },
                    "water": {
                        "value": 0.0,  # Missing water
                        "unit": "m³"
                    }
                }
            },
            "metadata": {
                "source_document": "SEWA_test.png"
            }
        }
        
        bill = SEWABill(**bill_data)
        assert bill.validation.confidence < 0.85  # Reduced due to missing water


class TestSchemaCompatibility:
    """Test compatibility with existing JSON schemas"""
    
    def test_dewa_schema_compatibility(self):
        """Test DEWA model matches existing schema structure"""
        # Load existing DEWA schema
        schema_path = Path(__file__).parent.parent.parent / "DEWA_Schema.json"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                existing_data = json.load(f)
                
            # Convert to Pydantic format
            converted_data = convert_existing_dewa_schema(existing_data)
            
            # Create DEWA bill from converted schema
            bill = DEWABill(**converted_data)
            
            # Verify key fields match
            assert bill.extracted_data.bill_info.account_number == existing_data["extractedData"]["billInfo"]["accountNumber"]
            assert bill.extracted_data.consumption_data.electricity.value == existing_data["extractedData"]["consumptionData"]["electricity"]["value"]
            
    def test_sewa_schema_compatibility(self):
        """Test SEWA model matches existing schema structure"""
        # Load existing SEWA schema
        schema_path = Path(__file__).parent.parent.parent / "SEWA_Schema.json"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                existing_data = json.load(f)
                
            # Convert to Pydantic format
            converted_data = convert_existing_sewa_schema(existing_data)
            
            # Create SEWA bill from converted schema
            bill = SEWABill(**converted_data)
            
            # Verify key fields match
            assert bill.extracted_data.bill_info.account_number == existing_data["extractedData"]["billInfo"]["accountNumber"]
            assert bill.extracted_data.consumption_data.electricity.value == existing_data["extractedData"]["consumptionData"]["electricity"]["value"]
            

if __name__ == "__main__":
    pytest.main([__file__, "-v"])