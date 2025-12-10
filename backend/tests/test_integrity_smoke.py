"""Smoke test for Data Integrity & Validation Framework."""

import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from quant.data.integrity import (
    DataValidator, OHLCVValidator, ValidationContext, ActionPolicy
)
from quant.data.integrity.models import ValidationReport, ValidationIssue


def test_valid_data():
    """Test validation with valid OHLCV data."""
    df = pd.DataFrame({
        'ticker': ['AAPL'] * 5,
        'date': pd.date_range('2024-01-01', periods=5),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'close': [104.0, 105.0, 106.0, 107.0, 108.0],
        'adj_close': [104.0, 105.0, 106.0, 107.0, 108.0],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    })
    
    validator = OHLCVValidator()
    report = validator.validate(df, context=ValidationContext.DAILY, lazy=True)
    
    print('Validation Report:')
    print(f'  Rows input: {report.rows_input}')
    print(f'  Rows output: {report.rows_output}')
    print(f'  Passed: {report.passed}')
    print(f'  Structural issues: {len(report.structural_issues)}')
    print(f'  Logical issues: {len(report.logical_issues)}')
    
    assert report.passed, "Valid data should pass validation"
    assert report.rows_input == 5
    assert len(report.structural_issues) == 0
    assert len(report.logical_issues) == 0
    print('✓ Valid data test passed\n')


def test_invalid_high_low():
    """Test validation with High < Low (invalid)."""
    invalid_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'date': pd.date_range('2024-01-01', periods=1),
        'open': [100.0],
        'high': [95.0],  # Invalid: High < Low
        'low': [99.0],
        'close': [98.0],
        'adj_close': [98.0],
        'volume': [1000000]
    })
    
    validator = OHLCVValidator()
    report = validator.validate(invalid_df, context=ValidationContext.DAILY, lazy=True)
    
    print('Invalid Data Report (High < Low):')
    print(f'  Logical issues: {len(report.logical_issues)}')
    for issue in report.logical_issues:
        print(f'    - {issue.message}')
    
    assert len(report.logical_issues) > 0, "Should detect High < Low"
    print('✓ Invalid High/Low test passed\n')


def test_eager_mode():
    """Test eager mode stops at first error."""
    invalid_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'date': pd.date_range('2024-01-01', periods=1),
        'open': [100.0],
        'high': [95.0],  # Invalid
        'low': [99.0],
        'close': [98.0],
        'adj_close': [98.0],
        'volume': [1000000]
    })
    
    validator = OHLCVValidator()
    report = validator.validate(invalid_df, context=ValidationContext.DAILY, lazy=False)
    
    print('Eager Mode Report:')
    print(f'  Logical issues: {len(report.logical_issues)}')
    print('✓ Eager mode test passed\n')


def test_missing_columns():
    """Test validation with missing required columns."""
    missing_col_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'date': pd.date_range('2024-01-01', periods=1),
        'open': [100.0],
    })
    
    validator = OHLCVValidator()
    report = validator.validate(missing_col_df, context=ValidationContext.DAILY, lazy=True)
    
    print('Missing Columns Report:')
    print(f'  Structural issues: {len(report.structural_issues)}')
    for issue in report.structural_issues:
        print(f'    - {issue.message}')
    
    assert len(report.structural_issues) > 0, "Should detect missing columns"
    print('✓ Missing columns test passed\n')


if __name__ == '__main__':
    test_valid_data()
    test_invalid_high_low()
    test_eager_mode()
    test_missing_columns()
    print('All smoke tests passed!')
