"""
Unit tests for POD OCR module.

Tests focus on regex parsing logic which can be validated without the OCR model.
These functions are copied here to avoid importing cv2/paddleocr dependencies.
"""

import pytest
import re
from typing import Optional


def parse_street_number(text: str) -> Optional[str]:
    """Parse street number from OCR text."""
    match = re.match(r'^(\d{1,5}[A-Z]?)$', text.strip(), re.IGNORECASE)
    return match.group(1) if match else None


def parse_unit_number(text: str) -> Optional[str]:
    """Parse unit/apartment number from OCR text."""
    match = re.search(r'(Apt|Unit|#|Suite)\s*(\d+[A-Z]?)', text, re.IGNORECASE)
    return match.group(0) if match else None


class TestParseStreetNumber:
    """Tests for street number parsing."""
    
    def test_single_digit(self):
        assert parse_street_number("1") == "1"
    
    def test_two_digits(self):
        assert parse_street_number("12") == "12"
    
    def test_three_digits(self):
        assert parse_street_number("123") == "123"
    
    def test_four_digits(self):
        assert parse_street_number("1234") == "1234"
    
    def test_five_digits(self):
        assert parse_street_number("12345") == "12345"
    
    def test_with_letter_suffix(self):
        assert parse_street_number("123A") == "123A"
        assert parse_street_number("1234B") == "1234B"
    
    def test_lowercase_letter_suffix(self):
        # Should still match (case insensitive)
        assert parse_street_number("123a") == "123a"
    
    def test_too_many_digits(self):
        # 6+ digits should not match typical street numbers
        assert parse_street_number("123456") is None
    
    def test_with_spaces(self):
        assert parse_street_number("  123  ") == "123"
    
    def test_invalid_format(self):
        assert parse_street_number("ABC") is None
        assert parse_street_number("12-34") is None
        assert parse_street_number("") is None


class TestParseUnitNumber:
    """Tests for unit/apartment number parsing."""
    
    def test_apt_format(self):
        assert parse_unit_number("Apt 1") == "Apt 1"
        assert parse_unit_number("Apt 12") == "Apt 12"
        assert parse_unit_number("Apt 1A") == "Apt 1A"
    
    def test_unit_format(self):
        assert parse_unit_number("Unit 5") == "Unit 5"
        assert parse_unit_number("Unit 100") == "Unit 100"
    
    def test_hash_format(self):
        assert parse_unit_number("#3") == "#3"
        assert parse_unit_number("# 42") == "# 42"
    
    def test_suite_format(self):
        assert parse_unit_number("Suite 200") == "Suite 200"
        assert parse_unit_number("Suite 10A") == "Suite 10A"
    
    def test_case_insensitive(self):
        assert parse_unit_number("APT 1") == "APT 1"
        assert parse_unit_number("apt 1") == "apt 1"
        assert parse_unit_number("UNIT 5") == "UNIT 5"
    
    def test_embedded_in_text(self):
        # Should find unit number within larger text
        result = parse_unit_number("Delivered to Apt 5B")
        assert result == "Apt 5B"
    
    def test_no_match(self):
        assert parse_unit_number("123 Main Street") is None
        assert parse_unit_number("Floor 2") is None
        assert parse_unit_number("") is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
