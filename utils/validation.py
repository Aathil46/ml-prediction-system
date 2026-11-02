"""Validation utilities for ShopperSense AI.
Handles file validation, schema checking, and data quality inspection.
"""

from typing import Dict, Any, Tuple, List, Optional
import io
import pandas as pd
import numpy as np

# Exact 17 customer session features expected prior to feature engineering
REQUIRED_COLUMNS = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay',
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend'
]

NUMERIC_COLUMNS = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
    'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'
]

VALID_MONTHS = {'Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Apr'}
VALID_VISITOR_TYPES = {'Returning_Visitor', 'New_Visitor', 'Other'}


def validate_uploaded_file(uploaded_file, max_size_mb: int = 50) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Validate file format, size, and parse into a pandas DataFrame.
    
    Args:
        uploaded_file: Uploaded file object from Streamlit or file-like buffer.
        max_size_mb: Maximum allowed file size in megabytes.
        
    Returns:
        Tuple of (is_valid, error_message, dataframe).
    """
    if uploaded_file is None:
        return False, "No file uploaded.", None

    # Check file size
    if hasattr(uploaded_file, 'size') and uploaded_file.size > max_size_mb * 1024 * 1024:
        return False, f"File exceeds maximum allowed size of {max_size_mb} MB.", None

    filename = getattr(uploaded_file, 'name', 'dataset.csv').lower()
    
    try:
        if filename.endswith('.csv'):
            # Try reading with utf-8, fallback to latin1
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
        elif filename.endswith(('.xlsx', '.xls')):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        else:
            return False, "Unsupported file format. Please upload a .csv, .xlsx, or .xls file.", None
            
        if df.empty:
            return False, "The uploaded file is empty (contains no data rows).", None
            
        return True, None, df
        
    except Exception as e:
        return False, f"Error parsing file: {str(e)}", None


def validate_input_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the uploaded DataFrame against the model's expected input schema.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        Dictionary with validation results: is_valid, missing_columns, extra_columns,
        has_revenue, errors, warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # Normalize column names (strip whitespace)
    df_cols = [str(c).strip() for c in df.columns]
    
    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_cols]
    if missing_cols:
        errors.append(
            f"Missing {len(missing_cols)} required column(s): {', '.join(missing_cols)}. "
            "Please ensure all required customer session features are present."
        )

    # Check for optional target column
    has_revenue = 'Revenue' in df_cols
    if has_revenue:
        warnings.append(
            "Detected 'Revenue' column. This column is treated as optional ground-truth for model "
            "evaluation and will be excluded from prediction inputs to prevent data leakage."
        )

    # Check for unexpected extra columns
    extra_cols = [col for col in df_cols if col not in REQUIRED_COLUMNS and col != 'Revenue']
    if extra_cols:
        warnings.append(
            f"Found {len(extra_cols)} extra column(s): {', '.join(extra_cols[:5])}" +
            ("..." if len(extra_cols) > 5 else "") + 
            ". These will be preserved in the final report but ignored during prediction."
        )

    return {
        "is_valid": len(errors) == 0,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "has_revenue": has_revenue,
        "errors": errors,
        "warnings": warnings
    }


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform data quality checks including missing values, duplicates, and valid value ranges.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary containing data health metrics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Missing values
    missing_series = df.isnull().sum()
    cols_with_missing = missing_series[missing_series > 0].to_dict()
    total_missing_cells = int(missing_series.sum())
    
    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0

    # Categorical validation
    invalid_months = []
    if 'Month' in df.columns:
        present_months = set(df['Month'].dropna().astype(str).unique())
        invalid_months = list(present_months - VALID_MONTHS)
        
    invalid_visitors = []
    if 'VisitorType' in df.columns:
        present_visitors = set(df['VisitorType'].dropna().astype(str).unique())
        invalid_visitors = list(present_visitors - VALID_VISITOR_TYPES)

    # Range checks for percentages/ratios
    range_warnings = []
    if 'BounceRates' in df.columns:
        invalid_bounce = ((df['BounceRates'] < 0.0) | (df['BounceRates'] > 1.0)).sum()
        if invalid_bounce > 0:
            range_warnings.append(f"{invalid_bounce} rows have 'BounceRates' outside the normal [0.0, 1.0] range.")
            
    if 'ExitRates' in df.columns:
        invalid_exit = ((df['ExitRates'] < 0.0) | (df['ExitRates'] > 1.0)).sum()
        if invalid_exit > 0:
            range_warnings.append(f"{invalid_exit} rows have 'ExitRates' outside the normal [0.0, 1.0] range.")

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_missing_cells": total_missing_cells,
        "cols_with_missing": cols_with_missing,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_pct, 2),
        "invalid_months": invalid_months,
        "invalid_visitors": invalid_visitors,
        "range_warnings": range_warnings,
        "is_healthy": total_missing_cells == 0 and len(invalid_months) == 0 and len(invalid_visitors) == 0
    }
