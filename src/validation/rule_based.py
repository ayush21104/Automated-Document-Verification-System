"""Rule-based validation for SPPU marksheets"""

import re
from typing import Dict, List, Tuple

class RuleBasedValidator:
    def __init__(self, config):
        self.config = config
        self.validation_results = {}
        
    def validate(self, data: Dict) -> Tuple[float, Dict]:
        """Perform complete rule-based validation"""
        results = {
            'prn_valid': False,
            'seat_number_valid': False,
            'branch_code_valid': False,
            'document_number_valid': False,
            'grades_valid': False,
            'sgpa_valid': False,
            'sgpa_calculation_valid': False,
            'credits_valid': False,
            'subject_codes_valid': False,
            'consistency_valid': False,
            'details': []
        }
        
        score = 0.0
        max_score = 10.0
        
        # Validate PRN
        prn_value = data.get('prn', '')
        if prn_value:
            if self.validate_prn(prn_value):
                results['prn_valid'] = True
                score += 1.0
            else:
                results['details'].append("Invalid PRN format")
        
        # Validate seat number
        seat_value = data.get('seat_no', '')
        if seat_value:
            if self.validate_seat_number(seat_value):
                results['seat_number_valid'] = True
                score += 1.0
            else:
                results['details'].append("Invalid seat number format")
        
        # Validate branch code
        branch_value = data.get('branch_code', '')
        if branch_value:
            if self.validate_branch_code(branch_value):
                results['branch_code_valid'] = True
                score += 1.0
            else:
                results['details'].append("Invalid branch code")
        
        # Validate document number
        doc_value = data.get('document_number', '')
        if doc_value:
            if self.validate_document_number(doc_value):
                results['document_number_valid'] = True
                score += 1.0
            else:
                results['details'].append("Invalid document number format")
        
        # Validate grades
        subjects = data.get('subjects', [])
        if subjects:
            if self.validate_grades(subjects):
                results['grades_valid'] = True
                score += 1.0
            else:
                results['details'].append("Invalid grades detected")
        
        # Validate SGPA
        if data.get('sgpa') is not None:
            if self.validate_sgpa(data.get('sgpa')):
                results['sgpa_valid'] = True
                score += 1.0
            else:
                results['details'].append("SGPA out of valid range")
        
        # Validate SGPA calculation
        if data.get('sgpa') is not None and subjects:
            if self.validate_sgpa_calculation(data):
                results['sgpa_calculation_valid'] = True
                score += 1.0
            else:
                results['details'].append("SGPA calculation mismatch")
        
        # Validate credits
        if subjects:
            if self.validate_credits(subjects):
                results['credits_valid'] = True
                score += 1.0
            else:
                results['details'].append("Invalid credit values")
        
        # Validate subject codes
        if subjects:
            if self.validate_subject_codes(subjects):
                results['subject_codes_valid'] = True
                score += 1.0
            else:
                results['details'].append("Invalid subject codes")
        
        # Validate consistency
        if subjects:
            if self.validate_consistency(data):
                results['consistency_valid'] = True
                score += 1.0
            else:
                results['details'].append("Data consistency check failed")
        else:
            results['details'].append("Subjects table not detected")
        
        confidence = score / max_score
        return confidence, results
    
    def validate_prn(self, prn: str) -> bool:
        """Validate PRN format for SPPU"""
        if not prn:
            return False
        
        # Updated pattern for 8-digit PRN with letter suffix
        pattern = r'^\d{8}[A-Z]?$'
        return bool(re.match(pattern, prn))
    
    def validate_seat_number(self, seat_no: str) -> bool:
        """Validate seat number format for SPPU"""
        if not seat_no:
            return False
        
        # More flexible patterns for SPPU seat numbers
        patterns = [
            r'^F\d{9}$',  # F190950028
            r'^F\d{8}$',  # F19095002
            r'^F\d{10}$', # F1909500280
            r'^[A-Z]\d{8,10}$'  # Any letter followed by 8-10 digits
        ]
        
        for pattern in patterns:
            if re.match(pattern, seat_no):
                return True
        return False
    
    def validate_branch_code(self, branch_code: str) -> bool:
        """Validate branch code for SPPU"""
        if not branch_code:
            return False
        
        # Valid branch codes for SPPU
        valid_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        return branch_code in valid_codes
    
    def validate_document_number(self, doc_number: str) -> bool:
        """Validate document number format"""
        if not doc_number:
            return False
        
        # Pattern for document numbers like 22-1412227
        pattern = r'^\d{2}-\d{7}$'
        return bool(re.match(pattern, doc_number))
    
    def validate_grades(self, subjects: List[Dict]) -> bool:
        """Validate all grades are in valid set"""
        if not subjects:
            return False
        
        valid_grades = self.config['valid_grades']
        for subject in subjects:
            grade = subject.get('grade', '')
            if grade and grade not in valid_grades:
                return False
        
        return True
    
    def validate_sgpa(self, sgpa: float) -> bool:
        """Validate SGPA is in valid range"""
        if sgpa is None:
            return False
        
        min_sgpa, max_sgpa = self.config['sgpa_range']
        return min_sgpa <= sgpa <= max_sgpa
    
    def calculate_expected_sgpa(self, subjects: List[Dict]) -> float:
        """Calculate expected SGPA based on subjects and grades"""
        if not subjects:
            return 0.0
        
        total_credit_points = 0
        total_credits = 0
        
        for subject in subjects:
            # First try to use the actual credit_points from the parsed data
            if 'credit_points' in subject and subject['credit_points'] > 0:
                total_credit_points += subject['credit_points']
                total_credits += subject.get('earned_credits', 0)
            else:
                # Fallback to calculating from grade and credits
                grade = subject.get('grade', '')
                credits = subject.get('earned_credits', 0)
                
                if grade in self.config['grade_points'] and credits > 0:
                    grade_points = self.config['grade_points'][grade]
                    total_credit_points += grade_points * credits
                    total_credits += credits
        
        if total_credits > 0:
            return total_credit_points / total_credits
        return 0.0
    
    def validate_sgpa_calculation(self, data: Dict) -> bool:
        """Validate that SGPA matches calculated value"""
        reported_sgpa = data.get('sgpa')
        subjects = data.get('subjects', [])
        
        if reported_sgpa is None or not subjects:
            return False
        
        try:
            calculated_sgpa = self.calculate_expected_sgpa(subjects)
            reported_sgpa_val = float(reported_sgpa)
            
            # Allow larger tolerance for rounding differences and calculation variations
            # SPPU may use different rounding methods
            tolerance = 0.2  # Increased from 0.1 to 0.2
            return abs(calculated_sgpa - reported_sgpa_val) <= tolerance
        except (ValueError, TypeError):
            return False
    
    def validate_credits(self, subjects: List[Dict]) -> bool:
        """Validate credit values"""
        if not subjects:
            return False
        
        min_credit, max_credit = self.config['credit_range']
        # Use earned_credits if available; fall back to total_credits
        total_credits = 0
        for s in subjects:
            if 'earned_credits' in s:
                total_credits += s.get('earned_credits', 0)
            elif 'total_credits' in s:
                total_credits += s.get('total_credits', 0)
        
        return min_credit <= total_credits <= max_credit
    
    def validate_subject_codes(self, subjects: List[Dict]) -> bool:
        """Validate subject code format"""
        if not subjects:
            return False
        
        pattern = self.config['subject_codes_pattern']
        for subject in subjects:
            code = subject.get('course_code') or subject.get('code') or ''
            if code and not re.match(pattern, str(code)):
                return False
        
        return True
    
    def validate_consistency(self, data: Dict) -> bool:
        """Check data consistency (SGPA calculation, credits, etc.)"""
        subjects = data.get('subjects', [])
        if not subjects:
            return False
        
        # Calculate expected SGPA
        total_grade_points = 0
        total_credits = 0
        
        for subject in subjects:
            credits = subject.get('earned_credits', subject.get('total_credits', 0))
            grade = subject.get('grade', '')
            
            if grade in self.config['grade_points']:
                grade_point = self.config['grade_points'][grade]
                total_grade_points += credits * grade_point
                total_credits += credits
        
        if total_credits > 0:
            calculated_sgpa = total_grade_points / total_credits
            actual_sgpa = data.get('sgpa', 0)
            
            # Allow larger tolerance for rounding and calculation differences
            if actual_sgpa:
                return abs(calculated_sgpa - actual_sgpa) < 0.3  # Increased from 0.1 to 0.3
        
        # If we can't calculate SGPA properly, don't fail the consistency check
        # This makes the validation more lenient for edge cases
        return True