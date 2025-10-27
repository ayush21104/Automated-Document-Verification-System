#!/usr/bin/env python3
"""
Test script to demonstrate SPPU marksheet verification capabilities
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_sppu_parsing():
    """Test SPPU marksheet parsing with sample data"""
    try:
        from src.ocr.text_extractor import SPPUMarksheetOCR
        from src.validation.rule_based import RuleBasedValidator
        from config import SPPU_CONFIG
        
        print("🧪 Testing SPPU Marksheet Parsing")
        print("=" * 50)
        
        # Initialize components
        ocr_engine = SPPUMarksheetOCR(SPPU_CONFIG)
        validator = RuleBasedValidator(SPPU_CONFIG)
        
        # Sample text from your marksheet
        sample_text = """
        No.: 22-1412227
        SAVITRIBAI PHULE PUNE UNIVERSITY
        (formerly University of Pune)
        GANESHKHIND PUNE 411 007
        STATEMENT OF MARKS/GRADES FOR F.E. (2019 CRED PAT) EXAM,OCT/NOV 2022
        BRANCH CODE: 05
        
        Seat No.: F190950028
        Centre: 12IT [95]
        Name: ARBAZ YUSUF SHAIKH
        Perm. Reg. No.: 72260330E
        Mother: RUBINA YUSUF SHAIKH
        College/School: [CEGP019110] - INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY
        
        COURSE CODE    COURSE NAME                    CO. TYPE  TOT. CRD  EARN. CRD  GRD  CRD. PTS
        102003: SYSTEMS IN MECH. ENGG. (* TH, 03, 03, B, 24)
        102003: SYSTEMS IN MECH. ENGG. (* PR, 01, 01, O, 10)
        104010: BASIC ELECTRONICS ENGG. (* TH, 03, 03, C, 21)
        104010: BASIC ELECTRONICS ENGG. (* PR, 01, 01, A, 09)
        107001: ENGINEERING MATHEMATICS I (* TH, 03, 03, A, 27)
        107001: ENGINEERING MATHEMATICS I (* TW, 01, 01, A, 09)
        107002: ENGINEERING PHYSICS (* TH, 04, 04, B, 32)
        107002: ENGINEERING PHYSICS (* PR, 01, 01, O, 10)
        110005: PROG. & PROBLEM SOLVING (* TH, 03, 03, B, 24)
        110005: PROG. & PROBLEM SOLVING (* PR, 01, 01, A, 09)
        111006: WORKSHOP (* PR, 01, 01, O, 10)
        101007: ENVIRONMENTAL STUDIES-I (* AC, 00, 00, AC, 00)
        
        SGPA1: 8.41
        TOTAL CREDITS EARNED: 22
        
        DATE: 05 MAY 2023
        R23052635905
        """
        
        # Parse the marksheet
        print("📄 Parsing marksheet structure...")
        parsed_data = ocr_engine.parse_marksheet_structure(sample_text)
        
        # Display parsed information
        print("\n📊 Parsed Information:")
        print(f"University: {parsed_data.get('university_name', 'N/A')}")
        print(f"Student Name: {parsed_data.get('student_name', 'N/A')}")
        print(f"PRN: {parsed_data.get('prn', 'N/A')}")
        print(f"Seat Number: {parsed_data.get('seat_no', 'N/A')}")
        print(f"Branch Code: {parsed_data.get('branch_code', 'N/A')}")
        print(f"Document Number: {parsed_data.get('document_number', 'N/A')}")
        print(f"SGPA: {parsed_data.get('sgpa', 'N/A')}")
        print(f"Total Credits Earned: {parsed_data.get('total_credits_earned', 'N/A')}")
        print(f"Exam Period: {parsed_data.get('exam_period', 'N/A')}")
        print(f"Result Date: {parsed_data.get('result_date', 'N/A')}")
        print(f"Reference Number: {parsed_data.get('reference_number', 'N/A')}")
        
        # Display subjects
        print(f"\n📚 Subjects ({len(parsed_data.get('subjects', []))}):")
        for i, subject in enumerate(parsed_data.get('subjects', []), 1):
            print(f"  {i}. {subject.get('course_code', 'N/A')}: {subject.get('course_name', 'N/A')}")
            print(f"     Type: {subject.get('course_type', 'N/A')}, Credits: {subject.get('earned_credits', 'N/A')}, Grade: {subject.get('grade', 'N/A')}")
        
        # Validate the parsed data
        print("\n🔍 Running validation...")
        validation_score, validation_results = validator.validate(parsed_data)
        
        print(f"\n✅ Validation Results:")
        print(f"Overall Score: {validation_score:.2%}")
        print(f"Validation Details:")
        
        for key, value in validation_results.items():
            if key != 'details':
                status = "✅" if value else "❌"
                print(f"  {status} {key.replace('_', ' ').title()}: {value}")
        
        if validation_results.get('details'):
            print(f"\n⚠️ Issues Found:")
            for detail in validation_results['details']:
                print(f"  • {detail}")
        
        # Test SGPA calculation
        print(f"\n🧮 SGPA Calculation Test:")
        calculated_sgpa = validator.calculate_expected_sgpa(parsed_data.get('subjects', []))
        reported_sgpa = parsed_data.get('sgpa', 0)
        print(f"Reported SGPA: {reported_sgpa}")
        print(f"Calculated SGPA: {calculated_sgpa:.2f}")
        print(f"Match: {'✅' if abs(calculated_sgpa - reported_sgpa) <= 0.1 else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎓 SPPU Document Verification System - Test")
    print("=" * 60)
    
    success = test_sppu_parsing()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test completed successfully!")
        print("\nThe system is now ready to verify SPPU marksheets with:")
        print("• Enhanced OCR parsing for detailed course information")
        print("• Comprehensive validation rules for SPPU-specific fields")
        print("• SGPA calculation and verification")
        print("• Support for theory/practical components")
        print("\nTo run the full application:")
        print("streamlit run main.py")
    else:
        print("❌ Test failed. Please check the errors above.")
        sys.exit(1)
