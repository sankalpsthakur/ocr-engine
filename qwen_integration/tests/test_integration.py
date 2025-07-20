"""Integration tests for end-to-end OCR pipeline"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_integration.src import OCRPipeline
from qwen_integration.src.models import DEWABill, SEWABill


class TestOCRPipeline:
    """Test the complete OCR pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        # Use a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        pipeline = OCRPipeline(temp_dir=Path(temp_dir))
        yield pipeline
        # Cleanup
        pipeline.cleanup()
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    @pytest.fixture
    def test_bills_dir(self):
        """Get test bills directory"""
        return Path(__file__).parent.parent.parent / "test_bills"
        
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert not pipeline.is_initialized
        pipeline.initialize()
        assert pipeline.is_initialized
        assert pipeline.qwen_processor.is_loaded
        
    def test_provider_detection(self, pipeline):
        """Test provider detection from filename and text"""
        # Test filename detection
        assert pipeline.detect_provider(Path("DEWA_bill.png")) == "DEWA"
        assert pipeline.detect_provider(Path("SEWA_2024.pdf")) == "SEWA"
        assert pipeline.detect_provider(Path("test_dewa.png")) == "DEWA"
        
        # Test text detection
        dewa_text = "Dubai Electricity and Water Authority\nAccount: 1234567890"
        assert pipeline.detect_provider(Path("bill.png"), dewa_text) == "DEWA"
        
        sewa_text = "Sharjah Electricity & Water Authority\nWater consumption: 100 m³"
        assert pipeline.detect_provider(Path("bill.png"), sewa_text) == "SEWA"
        
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "test_bills" / "DEWA.png").exists(),
        reason="Test bills not available"
    )
    def test_process_dewa_bill(self, pipeline, test_bills_dir):
        """Test processing DEWA bill"""
        dewa_file = test_bills_dir / "DEWA.png"
        
        if dewa_file.exists():
            result = pipeline.process_bill(dewa_file)
            
            # Check result type
            assert isinstance(result, DEWABill)
            
            # Check provider
            assert "DEWA" in result.extracted_data.bill_info.provider_name
            
            # Check required fields are present
            assert result.extracted_data.bill_info.account_number
            assert result.extracted_data.consumption_data.electricity.value > 0
            
            # Check metadata
            assert result.metadata.source_document == "DEWA.png"
            assert result.metadata.processing_time_seconds > 0
            
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "test_bills" / "SEWA.png").exists(),
        reason="Test bills not available"
    )
    def test_process_sewa_bill(self, pipeline, test_bills_dir):
        """Test processing SEWA bill"""
        sewa_file = test_bills_dir / "SEWA.png"
        
        if sewa_file.exists():
            result = pipeline.process_bill(sewa_file)
            
            # Check result type
            assert isinstance(result, SEWABill)
            
            # Check provider
            assert "SEWA" in result.extracted_data.bill_info.provider_name
            
            # Check both utilities are present
            assert result.extracted_data.consumption_data.electricity.value > 0
            assert result.extracted_data.consumption_data.water is not None
            
    def test_process_nonexistent_file(self, pipeline):
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            pipeline.process_bill("nonexistent_file.png")
            
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "test_bills").exists(),
        reason="Test bills directory not available"
    )
    def test_batch_processing(self, pipeline, test_bills_dir):
        """Test batch processing of multiple bills"""
        test_files = []
        
        # Find available test files
        for pattern in ["DEWA*.png", "SEWA*.png"]:
            test_files.extend(list(test_bills_dir.glob(pattern))[:2])
            
        if test_files:
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir)
                
                results = pipeline.process_batch(test_files, output_dir=output_path)
                
                # Check results
                assert len(results) == len(test_files)
                
                # Check output files were created
                output_files = list(output_path.glob("*_extracted.json"))
                assert len(output_files) == len([r for r in results if r is not None])
                
                # Verify JSON files are valid
                for json_file in output_files:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        assert "document_type" in data
                        assert data["document_type"] == "utility_bill"
                        

class TestSuryaExtractor:
    """Test Surya OCR extractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance"""
        from qwen_integration.src.extractors import SuryaExtractor
        temp_dir = tempfile.mkdtemp()
        extractor = SuryaExtractor(temp_dir=Path(temp_dir))
        yield extractor
        extractor.cleanup()
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    def test_postprocessing(self, extractor):
        """Test OCR text post-processing"""
        raw_text = """
        Dubai Electricity & Water Authority
        Account: O123456789O
        Consumption: 3OO kWn
        Date: 15 / 02 / 2024
        """
        
        processed = extractor._postprocess_text(raw_text)
        
        # Check corrections
        assert "kWh" in processed  # kWn -> kWh
        assert "Dubai Electricity and Water Authority" in processed  # & -> and
        assert "15/02/2024" in processed  # Fixed date spacing
        
    def test_validation(self, extractor):
        """Test extraction validation"""
        dewa_text = """
        Dubai Electricity and Water Authority (DEWA)
        Account Number: 1234567890
        Bill Date: 15/01/2024
        Electricity Consumption: 500 kWh
        CO2 Emissions: 200 kg CO2e
        """
        
        validations = extractor.validate_extraction(dewa_text, "DEWA")
        
        assert validations['has_account_number']
        assert validations['has_date']
        assert validations['has_electricity']
        assert validations['has_provider_name']
        assert validations['has_emissions']
        assert validations['overall_valid']
        
        
class TestQwenProcessor:
    """Test Qwen processor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        from qwen_integration.src.extractors import QwenProcessor
        # Use CPU for testing to avoid GPU requirements
        processor = QwenProcessor(device="cpu")
        yield processor
        processor.unload_model()
        
    def test_json_extraction(self, processor):
        """Test JSON extraction from model response"""
        # Test valid JSON extraction
        response = 'Here is the data: {"account_number": "1234567890", "bill_date": "15/01/2024"}'
        json_str = processor._extract_json(response)
        assert json_str == '{"account_number": "1234567890", "bill_date": "15/01/2024"}'
        
        # Test when JSON is the whole response
        response = '{"provider_name": "DEWA", "electricity_consumption": 500}'
        json_str = processor._extract_json(response)
        assert json_str == response
        
    def test_number_parsing(self, processor):
        """Test number parsing from various formats"""
        assert processor._parse_number(123.45) == 123.45
        assert processor._parse_number("123.45") == 123.45
        assert processor._parse_number("123.45 kWh") == 123.45
        assert processor._parse_number("AED 123.45") == 123.45
        assert processor._parse_number("") == 0.0
        assert processor._parse_number(None) == 0.0
        
    def test_date_parsing(self, processor):
        """Test date parsing to DD/MM/YYYY format"""
        assert processor._parse_date("15/01/2024") == "15/01/2024"
        assert processor._parse_date("2024-01-15") == "15/01/2024"
        assert processor._parse_date("15-01-2024") == "15/01/2024"
        assert processor._parse_date("15.01.2024") == "15/01/2024"
        assert processor._parse_date("") == ""
        assert processor._parse_date("invalid") == ""
        
    def test_fallback_extraction(self, processor):
        """Test fallback extraction using regex"""
        ocr_text = """
        Dubai Electricity and Water Authority
        Account Number: 1234567890
        Bill Date: 15/01/2024
        Electricity Consumption: 500 kWh
        """
        
        data = processor._fallback_extraction(ocr_text, "DEWA")
        
        assert data['account_number'] == "1234567890"
        assert data['bill_date'] == "15/01/2024"
        assert data['electricity_consumption'] == 500.0
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])