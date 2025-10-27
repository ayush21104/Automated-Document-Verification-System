"""Database module for storing verification history"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List
import pandas as pd

class VerificationDatabase:
    def __init__(self, config):
        self.config = config
        self.db_path = config.get('db_path', 'verification_records.db')
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prn TEXT,
                student_name TEXT,
                verification_status TEXT,
                confidence_score REAL,
                ocr_confidence REAL,
                rule_based_score REAL,
                cnn_score REAL,
                isolation_forest_score REAL,
                nlp_score REAL,
                details TEXT,
                raw_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_verification_result(self, result: Dict):
        """Save verification result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO verification_history 
            (prn, student_name, verification_status, confidence_score,
             ocr_confidence, rule_based_score, cnn_score, 
             isolation_forest_score, nlp_score, details, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('prn', ''),
            result.get('student_name', ''),
            result.get('verification_status', ''),
            result.get('confidence_score', 0),
            result.get('ocr_confidence', 0),
            result.get('rule_based_score', 0),
            result.get('cnn_score', 0),
            result.get('isolation_forest_score', 0),
            result.get('nlp_score', 0),
            json.dumps(result.get('details', {})),
            json.dumps(result.get('raw_data', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def get_verification_history(self, limit: int = 100) -> pd.DataFrame:
        """Get verification history"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM verification_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def get_statistics(self) -> Dict:
        """Get verification statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total verifications
        cursor.execute('SELECT COUNT(*) FROM verification_history')
        stats['total_verifications'] = cursor.fetchone()[0]
        
        # Status distribution
        cursor.execute('''
            SELECT verification_status, COUNT(*) 
            FROM verification_history 
            GROUP BY verification_status
        ''')
        stats['status_distribution'] = dict(cursor.fetchall())
        
        # Average confidence scores
        cursor.execute('''
            SELECT 
                AVG(confidence_score) as avg_confidence,
                AVG(ocr_confidence) as avg_ocr,
                AVG(rule_based_score) as avg_rule,
                AVG(cnn_score) as avg_cnn,
                AVG(isolation_forest_score) as avg_isolation,
                AVG(nlp_score) as avg_nlp
            FROM verification_history
        ''')
        
        result = cursor.fetchone()
        stats['average_scores'] = {
            'overall': result[0],
            'ocr': result[1],
            'rule_based': result[2],
            'cnn': result[3],
            'isolation_forest': result[4],
            'nlp': result[5]
        }
        
        conn.close()
        return stats