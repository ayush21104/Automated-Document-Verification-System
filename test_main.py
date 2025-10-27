#!/usr/bin/env python3
"""
Test script to verify main.py can be imported and components initialized
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test if all imports work correctly"""
    try:
        print("Testing imports...")
        
        # Test config import
        from config import SPPU_CONFIG, VALIDATION_THRESHOLDS, CNN_CONFIG
        print("✅ Config imports successful")
        
        # Test module imports
        from src.preprocessing.image_processor import ImagePreprocessor
        from src.ocr.text_extractor import SPPUMarksheetOCR
        from src.validation.nlp_validator import NLPValidator
        from src.validation.rule_based import RuleBasedValidator
        from src.validation.statistical import CNNForgeryDetector
        from src.ml_models.isolation_forest import AnomalyDetector
        from src.utils.database import VerificationDatabase
        from src.utils.scoring import ConfidenceScorer
        from src.utils.visualization import visualize_results
        from src.utils.utils import load_image, save_image
        print("✅ All module imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_component_initialization():
    """Test if components can be initialized"""
    try:
        print("\nTesting component initialization...")
        
        from config import SPPU_CONFIG, VALIDATION_THRESHOLDS, CNN_CONFIG, ISOLATION_FOREST_CONFIG, NLP_CONFIG, DB_CONFIG
        from src.preprocessing.image_processor import ImagePreprocessor
        from src.ocr.text_extractor import SPPUMarksheetOCR
        from src.validation.nlp_validator import NLPValidator
        from src.validation.rule_based import RuleBasedValidator
        from src.validation.statistical import CNNForgeryDetector
        from src.ml_models.isolation_forest import AnomalyDetector
        from src.utils.database import VerificationDatabase
        from src.utils.scoring import ConfidenceScorer
        
        # Initialize components
        image_processor = ImagePreprocessor()
        print("✅ ImagePreprocessor initialized")
        
        ocr_engine = SPPUMarksheetOCR(SPPU_CONFIG)
        print("✅ SPPUMarksheetOCR initialized")
        
        nlp_validator = NLPValidator(NLP_CONFIG)
        print("✅ NLPValidator initialized")
        
        rule_validator = RuleBasedValidator(SPPU_CONFIG)
        print("✅ RuleBasedValidator initialized")
        
        cnn_detector = CNNForgeryDetector(CNN_CONFIG)
        print("✅ CNNForgeryDetector initialized")
        
        anomaly_detector = AnomalyDetector(ISOLATION_FOREST_CONFIG)
        print("✅ AnomalyDetector initialized")
        
        database = VerificationDatabase(DB_CONFIG)
        print("✅ VerificationDatabase initialized")
        
        scorer = ConfidenceScorer(VALIDATION_THRESHOLDS)
        print("✅ ConfidenceScorer initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_import():
    """Test if main.py can be imported"""
    try:
        print("\nTesting main.py import...")
        
        # Import main module
        import main
        print("✅ main.py imported successfully")
        
        # Test if main function exists
        if hasattr(main, 'main'):
            print("✅ main() function found")
        else:
            print("❌ main() function not found")
            return False
            
        if hasattr(main, 'initialize_components'):
            print("✅ initialize_components() function found")
        else:
            print("❌ initialize_components() function not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ main.py import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Document Verification System")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_component_initialization():
        tests_passed += 1
    
    if test_main_import():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The system is ready to run.")
        print("\nTo run the application:")
        print("streamlit run main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

