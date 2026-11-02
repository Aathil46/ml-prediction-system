"""Preprocessing module for ShopperSense AI.
Implements the exact feature engineering, one-hot encoding, and feature alignment
expected by the pretrained Random Forest Classifier.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Exact 30 feature names and order as verified from the pretrained Random Forest artifact
EXPECTED_MODEL_FEATURES = [
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
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'Weekend',
    'Total_Pages',
    'Total_Duration',
    'Avg_Duration_Per_Page',
    'Bounce_Exit_Avg',
    'Month_Dec',
    'Month_Feb',
    'Month_Jul',
    'Month_June',
    'Month_Mar',
    'Month_May',
    'Month_Nov',
    'Month_Oct',
    'Month_Sep',
    'VisitorType_Other',
    'VisitorType_Returning_Visitor'
]

# Encoded categorical feature lists
EXPECTED_MONTH_DUMMIES = [
    'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar',
    'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep'
]

EXPECTED_VISITOR_DUMMIES = [
    'VisitorType_Other', 'VisitorType_Returning_Visitor'
]


def preprocess_features(df_input: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform raw customer session data into the exact 30-feature matrix expected by Random Forest.
    
    Steps:
    1. Clone DataFrame and isolate optional Revenue target.
    2. Convert Weekend flag to integer (0 or 1).
    3. Calculate interaction features:
       - Total_Pages = Administrative + Informational + ProductRelated
       - Total_Duration = Administrative_Duration + Informational_Duration + ProductRelated_Duration
       - Avg_Duration_Per_Page = Total_Duration / (Total_Pages + 1e-5)
       - Bounce_Exit_Avg = (BounceRates + ExitRates) / 2.0
    4. One-hot encode Month and VisitorType using identical drop_first schema
       (reference categories: Month='Aug', VisitorType='New_Visitor').
    5. Align and order features exactly to match EXPECTED_MODEL_FEATURES.
    
    Args:
        df_input: Raw input DataFrame with the 17 customer session features.
        
    Returns:
        Tuple of (X_preprocessed DataFrame [N x 30], transformation_summary dict).
    """
    df = df_input.copy()

    # Drop target column if present to ensure zero leakage
    if 'Revenue' in df.columns:
        df = df.drop(columns=['Revenue'])

    # Handle Weekend conversion robustly
    if 'Weekend' in df.columns:
        if df['Weekend'].dtype == bool or df['Weekend'].dtype == object:
            df['Weekend'] = df['Weekend'].map({
                True: 1, False: 0,
                'True': 1, 'False': 0,
                'true': 1, 'false': 0,
                'TRUE': 1, 'FALSE': 0,
                1: 1, 0: 0, '1': 1, '0': 0
            }).fillna(0).astype(int)
        else:
            df['Weekend'] = df['Weekend'].fillna(0).astype(int)

    # Ensure numeric columns are strictly numeric
    numeric_cols = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
        'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Feature Engineering
    df['Total_Pages'] = (
        df['Administrative'] + df['Informational'] + df['ProductRelated']
    )
    df['Total_Duration'] = (
        df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    )
    df['Avg_Duration_Per_Page'] = df['Total_Duration'] / (df['Total_Pages'] + 1e-5)
    df['Bounce_Exit_Avg'] = (df['BounceRates'] + df['ExitRates']) / 2.0

    # 2. Categorical One-Hot Encoding
    # Month: Reference month is 'Aug' (all dummy indicators = 0)
    month_series = df['Month'].astype(str).str.strip()
    for dummy_col in EXPECTED_MONTH_DUMMIES:
        target_month = dummy_col.replace('Month_', '')
        df[dummy_col] = (month_series == target_month).astype(int)

    # VisitorType: Reference category is 'New_Visitor' (all dummy indicators = 0)
    visitor_series = df['VisitorType'].astype(str).str.strip()
    for dummy_col in EXPECTED_VISITOR_DUMMIES:
        target_visitor = dummy_col.replace('VisitorType_', '')
        df[dummy_col] = (visitor_series == target_visitor).astype(int)

    # 3. Ensure all 30 expected features exist and match exact order
    for col in EXPECTED_MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X_preprocessed = df[EXPECTED_MODEL_FEATURES].copy()

    # Create summary metrics of engineered features for business transparency
    summary = {
        "rows_processed": len(X_preprocessed),
        "features_count": len(EXPECTED_MODEL_FEATURES),
        "avg_total_pages": float(df['Total_Pages'].mean()),
        "avg_total_duration_sec": float(df['Total_Duration'].mean()),
        "avg_duration_per_page_sec": float(df['Avg_Duration_Per_Page'].mean()),
        "avg_bounce_exit_score": float(df['Bounce_Exit_Avg'].mean()),
        "engineered_columns": ['Total_Pages', 'Total_Duration', 'Avg_Duration_Per_Page', 'Bounce_Exit_Avg'],
        "encoded_categorical_columns": EXPECTED_MONTH_DUMMIES + EXPECTED_VISITOR_DUMMIES
    }

    return X_preprocessed, summary
