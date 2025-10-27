"""OCR module for text extraction from marksheets"""

import pytesseract
import cv2
import re
import pandas as pd
from typing import Dict, List
import numpy as np
from config import TESSERACT_CONFIG, OCR_LANGUAGES


class SPPUMarksheetOCR:
    def __init__(self, config: Dict):
        self.config = config
        self.extracted_data = {}

    # ------------------- OCR extraction -------------------

    def extract_text(self, image) -> str:
        """Extract raw text from image using Tesseract OCR"""
        try:
            if image is None:
                return ""
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            text = pytesseract.image_to_string(
                rgb, lang=OCR_LANGUAGES, config=TESSERACT_CONFIG
            )
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def extract_with_confidence(self, image) -> pd.DataFrame:
        """Extract text with confidence scores"""
        try:
            if image is None:
                return pd.DataFrame()
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data = pytesseract.image_to_data(
                rgb,
                lang=OCR_LANGUAGES,
                config=TESSERACT_CONFIG,
                output_type=pytesseract.Output.DATAFRAME,
            )
            data = data[data["text"].notna() & (data["text"] != "")]
            return data
        except Exception as e:
            print(f"OCR with confidence error: {e}")
            return pd.DataFrame()

    def calculate_confidence(self, ocr_data: pd.DataFrame) -> float:
        """Calculate overall OCR confidence score"""
        try:
            if ocr_data.empty:
                return 0.0

            confidences = ocr_data[ocr_data["conf"] > 0]["conf"]
            if confidences.empty:
                return 0.0

            weights = ocr_data[ocr_data["conf"] > 0]["text"].str.len()
            weighted_conf = np.average(confidences, weights=weights)
            return weighted_conf / 100.0
        except Exception as e:
            print(f"Error calculating OCR confidence: {e}")
            return 0.0

    # ------------------- Helper: Group words into lines -------------------

    def _ocr_dataframe_to_lines(self, image) -> List[Dict]:
        """Group OCR dataframe into line dictionaries"""
        try:
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ocr_df = pytesseract.image_to_data(
                rgb,
                lang=OCR_LANGUAGES,
                config=TESSERACT_CONFIG,
                output_type=pytesseract.Output.DATAFRAME,
            )
        except Exception as e:
            print("image_to_data error:", e)
            return []

        ocr_df = ocr_df[ocr_df["text"].notna() & (ocr_df["text"].str.strip() != "")]
        if ocr_df.empty:
            return []

        grouped = ocr_df.groupby(
            ["page_num", "block_num", "par_num", "line_num"], sort=False
        )

        lines = []
        for (_, _, _, _), g in grouped:
            g_sorted = g.sort_values("left")
            words = []
            for _, row in g_sorted.iterrows():
                words.append(
                    {
                        "text": str(row["text"]).strip(),
                        "conf": float(row["conf"])
                        if "conf" in row and not pd.isna(row["conf"])
                        else -1,
                        "left": int(row["left"]),
                        "top": int(row["top"]),
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                    }
                )
            line_text = " ".join([w["text"] for w in words])
            lines.append({"text": line_text, "words": words})
        return lines

    # ------------------- Main parser -------------------

    def parse_marksheet_structure(self, text_or_image) -> Dict:
        """Parse marksheet into structured JSON"""
        data = {
            "university_name": "",
            "prn": "",
            "student_name": "",
            "mother_name": "",
            "college_name": "",
            "seat_no": "",
            "center": "",
            "branch": "",
            "branch_code": "",
            "exam_date": "",
            "exam_period": "",
            "document_number": "",
            "subjects": [],
            "sgpa": None,
            "total_credits": 0,
            "total_credits_earned": 0,
            "result_date": "",
            "medium_of_instruction": "",
            "reference_number": "",
            "signatory": "",
        }

        # Lines from either OCR dataframe or raw text
        if not isinstance(text_or_image, str):
            lines = self._ocr_dataframe_to_lines(text_or_image)
            raw_lines = [l["text"] for l in lines]
        else:
            raw_lines = text_or_image.split("\n")

        # University name
        for ln in raw_lines[:10]:
            if "savitri" in ln.lower() and "pune" in ln.lower():
                data["university_name"] = ln.strip()
                break

        # PRN
        for ln in raw_lines:
            m = re.search(r"(?:PRN|Perm\. Reg\. No)\s*[:\-]?\s*([A-Z0-9]+)", ln, re.I)
            if m:
                data["prn"] = m.group(1)
                break

        # Seat No
        for ln in raw_lines:
            m = re.search(r"Seat\s*No[:\-]?\s*([A-Z0-9]+)", ln, re.I)
            if m:
                data["seat_no"] = m.group(1)
                break

        # Student & Mother Name
        for ln in raw_lines:
            s = re.search(r"Student Name[:\-]?\s*(.+)", ln, re.I)
            if s:
                data["student_name"] = s.group(1).strip()
            m = re.search(r"Mother Name[:\-]?\s*(.+)", ln, re.I)
            if m:
                data["mother_name"] = m.group(1).strip()

        # SGPA
        for ln in raw_lines:
            if "sgpa" in ln.lower():
                m = re.search(r"([0-9]+\.[0-9]+)", ln)
                if m:
                    data["sgpa"] = float(m.group(1))
                    break

        # Total Credits Earned
        for ln in raw_lines:
            if "total credits" in ln.lower():
                m = re.search(r"(\d+)", ln)
                if m:
                    data["total_credits_earned"] = int(m.group(1))
                    break

        # Result Date
        for ln in raw_lines:
            m = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})", ln)
            if m:
                data["result_date"] = m.group(1)
                break

        # Subjects
        data["subjects"] = self._parse_subjects_from_lines(raw_lines)

        return data

    # ------------------- Subject parsing -------------------

    def _parse_subjects_from_lines(self, lines: List[str]) -> List[Dict]:
        subjects = []
        started = False
        for ln in lines:
            text = ln.strip()
            if re.match(r"^\d{5,6}\b", text):
                started = True
            if started and (
                "sgpa" in text.lower()
                or "total credits" in text.lower()
                or "result" in text.lower()
            ):
                break
            if started and text:
                sub = self._parse_subject_line_table(text)
                if sub:
                    subjects.append(sub)
        return subjects

    def _parse_subject_line_table(self, line: str) -> Dict:
        """Parse table-like subject rows"""
        # Try to parse SPPU format: 102003: SYSTEMS IN MECH. ENGG. (* TH, 03, 03, B, 24)
        sppu_pattern = r"(\d{5,6}):\s+(.+?)\s+\(\*\s*([A-Z]{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*([A-Z]{1,3}),\s*(\d{1,3})\)"
        m = re.search(sppu_pattern, line)
        if m:
            return {
                "course_code": m.group(1),
                "course_name": m.group(2).strip(),
                "course_type": m.group(3),
                "total_credits": int(m.group(4)),
                "earned_credits": int(m.group(5)),
                "grade": m.group(6),
                "credit_points": int(m.group(7)),
            }
        
        # Try simple table format
        parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
        if len(parts) >= 4 and re.match(r"^\d{5,6}$", parts[0]):
            course_code = parts[0]
            course_name = parts[1]
            credits = int(re.sub(r"\D", "", parts[2])) if parts[2].isdigit() else 0
            grade = parts[3]
            credit_points = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
            return {
                "course_code": course_code,
                "course_name": course_name,
                "course_type": "",
                "total_credits": credits,
                "earned_credits": credits,
                "grade": grade,
                "credit_points": credit_points,
            }

        # Try simple regex format
        m = re.search(
            r"^(\d{5,6})\s+(.+?)\s+(\d{1,2})\s+([A-Z]{1,3})\s+(\d{1,3})$", line
        )
        if m:
            return {
                "course_code": m.group(1),
                "course_name": m.group(2).strip(),
                "course_type": "",
                "total_credits": int(m.group(3)),
                "earned_credits": int(m.group(3)),
                "grade": m.group(4),
                "credit_points": int(m.group(5)),
            }
        return None
