"""NLP-based validation for text consistency and semantic analysis"""

import spacy
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from typing import Dict, List, Tuple
import re

class NLPValidator:
    def __init__(self, config):
        self.config = config
        self.nlp = None
        self.load_models()
        
    def load_models(self):
        """Load NLP models"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Load spaCy model
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Warning: NLP models not fully loaded. Install with: python -m spacy download en_core_web_sm")
    
    def validate_text_consistency(self, data: Dict) -> Tuple[float, Dict]:
        """Validate text consistency and semantic correctness"""
        results = {
            'university_valid': False,
            'subject_names_valid': False,
            'semantic_consistency': False,
            'text_quality': 0.0,
            'details': []
        }
        
        score = 0.0
        max_score = 4.0
        
        # Validate university name
        if self.validate_university_name(data.get('university_name', '')):
            results['university_valid'] = True
            score += 1.0
        else:
            results['details'].append("University name mismatch")
        
        # Validate subject names
        if self.validate_subject_names(data.get('subjects', [])):
            results['subject_names_valid'] = True
            score += 1.0
        else:
            results['details'].append("Suspicious subject names detected")
        
        # Check semantic consistency
        if self.check_semantic_consistency(data):
            results['semantic_consistency'] = True
            score += 1.0
        else:
            results['details'].append("Semantic inconsistencies found")
        
        # Assess text quality
        text_quality = self.assess_text_quality(data.get('raw_text', ''))
        results['text_quality'] = text_quality
        if text_quality > 0.7:
            score += 1.0
        else:
            results['details'].append("Poor text quality detected")
        
        confidence = score / max_score
        return confidence, results
    
    def validate_university_name(self, university_name: str) -> bool:
        """Validate university name using fuzzy matching"""
        if not university_name:
            return False
        
        expected = self.config.get('university_name', 'SAVITRIBAI PHULE PUNE UNIVERSITY')
        
        # Clean and normalize
        uni_clean = university_name.upper().strip()
        expected_clean = expected.upper()
        
        # Check for key terms
        key_terms = ['SAVITRIBAI', 'PHULE', 'PUNE', 'UNIVERSITY']
        matches = sum(1 for term in key_terms if term in uni_clean)
        
        return matches >= 3
    
    def validate_subject_names(self, subjects: List[Dict]) -> bool:
        """Validate subject names against known patterns"""
        if not subjects:
            return False
        
        valid_keywords = self.config.get('subject_keywords', [])
        suspicious_count = 0
        
        for subject in subjects:
            name = subject.get('course_name', subject.get('name', '')).lower()
            if not name:
                continue
            
            # Check if subject name contains valid keywords
            has_valid_keyword = any(keyword in name for keyword in valid_keywords)
            
            # Check for suspicious patterns
            if not has_valid_keyword and len(name) > 3:
                # Check if it's mostly gibberish
                if not self.is_valid_text(name):
                    suspicious_count += 1
        
        # Allow up to 20% suspicious subjects
        return suspicious_count < len(subjects) * 0.2
    
    def check_semantic_consistency(self, data: Dict) -> bool:
        """Check semantic consistency across document"""
        if not self.nlp:
            return True  # Skip if NLP not loaded
        
        # Extract key information
        student_name = data.get('student_name', '')
        mother_name = data.get('mother_name', '')
        
        # Basic checks
        if student_name and mother_name:
            # Names should be proper nouns
            if not self.is_proper_name(student_name):
                return False
            if not self.is_proper_name(mother_name):
                return False
        
        # Check subject coherence
        subjects = data.get('subjects', [])
        if subjects:
            # All subjects should be academic in nature
            for subject in subjects:
                name = subject.get('course_name', subject.get('name', ''))
                if name and not self.is_academic_subject(name):
                    return False
        
        return True
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text is valid (not gibberish)"""
        if not text:
            return False
        
        # Check for minimum vowels
        vowels = set('aeiouAEIOU')
        vowel_count = sum(1 for char in text if char in vowels)
        
        # Text should have reasonable vowel ratio
        if len(text) > 3:
            vowel_ratio = vowel_count / len(text)
            if vowel_ratio < 0.15 or vowel_ratio > 0.7:
                return False
        
        # Check for repeated characters
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                return False
        
        return True
    
    def is_proper_name(self, name: str) -> bool:
        """Check if text is a proper name"""
        if not name:
            return False
        
        # Names should start with capital letters
        words = name.split()
        for word in words:
            if word and not word[0].isupper():
                return False
        
        # Names shouldn't contain numbers or special characters
        if re.search(r'[0-9@#$%^&*()_+=\[\]{};:"\\|,.<>?/]', name):
            return False
        
        return True
    
    def is_academic_subject(self, subject_name: str) -> bool:
        """Check if subject name is academic"""
        academic_keywords = [
            'database', 'computer', 'programming', 'system', 'network',
            'theory', 'practice', 'laboratory', 'lab', 'management',
            'engineering', 'mathematics', 'science', 'technology',
            'seminar', 'project', 'design', 'analysis', 'study'
        ]
        
        name_lower = subject_name.lower()
        return any(keyword in name_lower for keyword in academic_keywords)
    
    def assess_text_quality(self, text: str) -> float:
        """Assess overall text quality"""
        if not text:
            return 0.0
        
        quality_score = 1.0
        
        # Check for excessive noise characters
        noise_chars = text.count('�') + text.count('□') + text.count('■')
        if noise_chars > len(text) * 0.05:
            quality_score -= 0.3
        
        # Check for proper word separation
        words = text.split()
        if len(words) < 50:  # Marksheet should have substantial text
            quality_score -= 0.2
        
        # Check for reasonable line breaks
        lines = text.split('\n')
        empty_lines = sum(1 for line in lines if not line.strip())
        if empty_lines > len(lines) * 0.5:
            quality_score -= 0.2
        
        return max(0.0, quality_score)