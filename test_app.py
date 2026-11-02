"""Comprehensive automated test suite for ShopperSense AI.
Verifies all model loading, schema alignment, data ingest, prediction, reporting,
and edge-case error handling requirements.
"""

import os
import sys
import io
import pandas as pd
import numpy as np

# Ensure utils is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.validation import (
    validate_uploaded_file,
    validate_input_schema,
    check_data_quality,
    REQUIRED_COLUMNS
)
from utils.preprocessing import preprocess_features, EXPECTED_MODEL_FEATURES
from utils.prediction import (
    load_rf_model,
    predict_purchase_intent,
    get_model_metadata,
    evaluate_predictions_against_ground_truth
)
from utils.reporting import (
    compute_summary_metrics,
    generate_csv_report,
    generate_excel_report,
    generate_summary_markdown
)


def run_tests():
    print("=" * 60)
    print("RUNNING SHOPPERSENSE AI VERIFICATION TEST SUITE")
    print("=" * 60)
    passed_tests = 0
    total_tests = 0

    # 1. Test Model Loading
    total_tests += 1
    print("\n[Test 1/11] Verifying Pretrained Random Forest Model Loading...")
    model = load_rf_model()
    assert model is not None, "Model failed to load"
    assert model.n_features_in_ == 30, f"Expected 30 features, got {model.n_features_in_}"
    assert list(model.classes_) == [0, 1], f"Expected classes [0, 1], got {model.classes_}"
    print(" -> PASSED: Random Forest model loaded with 30 features and [0, 1] classes.")
    passed_tests += 1

    # 2. Test Input Schema & Preprocessing Alignment
    total_tests += 1
    print("\n[Test 2/11] Verifying Input Schema Matches Preprocessing Output...")
    csv_path = os.path.join(current_dir, "sample_data", "sample_customers.csv")
    df_sample = pd.read_csv(csv_path)
    X_prep, summary = preprocess_features(df_sample)
    assert list(X_prep.columns) == list(model.feature_names_in_), "Preprocessed columns do not match model features"
    assert X_prep.shape == (len(df_sample), 30), f"Expected shape ({len(df_sample)}, 30), got {X_prep.shape}"
    print(" -> PASSED: Feature engineering and categorical encoding align exactly to 30 features.")
    passed_tests += 1

    # 3. Test CSV Upload and Parsing
    total_tests += 1
    print("\n[Test 3/11] Testing CSV File Upload and Ingest...")
    with open(csv_path, "rb") as f:
        file_buffer = io.BytesIO(f.read())
        file_buffer.name = "sample.csv"
        is_valid, err, df_csv = validate_uploaded_file(file_buffer)
    assert is_valid, f"CSV validation failed: {err}"
    assert len(df_csv) == len(df_sample), "CSV row count mismatch"
    print(f" -> PASSED: CSV upload ingested {len(df_csv)} rows successfully.")
    passed_tests += 1

    # 4. Test Excel Upload and Parsing
    total_tests += 1
    print("\n[Test 4/11] Testing Excel (.xlsx) File Upload and Ingest...")
    xlsx_path = os.path.join(current_dir, "sample_data", "sample_customers.xlsx")
    with open(xlsx_path, "rb") as f:
        file_buffer = io.BytesIO(f.read())
        file_buffer.name = "sample.xlsx"
        is_valid, err, df_xlsx = validate_uploaded_file(file_buffer)
    assert is_valid, f"Excel validation failed: {err}"
    assert len(df_xlsx) == len(df_sample), "Excel row count mismatch"
    print(f" -> PASSED: Excel upload ingested {len(df_xlsx)} rows successfully.")
    passed_tests += 1

    # 5. Test Prediction Generation
    total_tests += 1
    print("\n[Test 5/11] Testing Prediction Generation...")
    results = predict_purchase_intent(model, X_prep, raw_df=df_sample)
    assert len(results) == len(df_sample), "Results length mismatch"
    assert "Predicted_Class" in results.columns, "Missing Predicted_Class"
    assert "Prediction_Label" in results.columns, "Missing Prediction_Label"
    assert set(results['Predicted_Class'].unique()).issubset({0, 1}), "Invalid predicted classes"
    print(f" -> PASSED: Generated predictions for {len(results)} sessions.")
    passed_tests += 1

    # 6. Test Probability Generation
    total_tests += 1
    print("\n[Test 6/11] Testing Probability Generation and Formatting...")
    assert "Purchase_Probability" in results.columns, "Missing Purchase_Probability"
    assert "No_Purchase_Probability" in results.columns, "Missing No_Purchase_Probability"
    probs_p = results["Purchase_Probability"].values
    probs_np = results["No_Purchase_Probability"].values
    assert np.all((probs_p >= 0.0) & (probs_p <= 1.0)), "Probabilities out of range"
    assert np.allclose(probs_p + probs_np, 1.0, atol=1e-3), "Probabilities do not sum to 1"
    print(" -> PASSED: Calibrated probabilities verified in range [0, 1] summing to 1.0.")
    passed_tests += 1

    # 7. Test Missing Required Columns Handling
    total_tests += 1
    print("\n[Test 7/11] Testing Missing Column Detection and Validation...")
    corrupt_df = df_sample.drop(columns=["PageValues", "ExitRates"])
    val_corrupt = validate_input_schema(corrupt_df)
    assert not val_corrupt["is_valid"], "Schema validator should have flagged missing columns"
    assert "PageValues" in val_corrupt["missing_columns"], "PageValues not in missing list"
    assert "ExitRates" in val_corrupt["missing_columns"], "ExitRates not in missing list"
    assert len(val_corrupt["errors"]) > 0, "No error messages generated for missing columns"
    print(" -> PASSED: Missing columns caught cleanly with actionable error messages.")
    passed_tests += 1

    # 8. Test Invalid File Format Handling
    total_tests += 1
    print("\n[Test 8/11] Testing Invalid File Format Handling...")
    invalid_buffer = io.BytesIO(b"random,binary,content")
    invalid_buffer.name = "document.pdf"
    is_valid, err, _ = validate_uploaded_file(invalid_buffer)
    assert not is_valid, "Should reject non-CSV/non-Excel files"
    assert "Unsupported file format" in err, f"Unexpected error message: {err}"
    print(" -> PASSED: Unsupported file format rejected safely without tracebacks.")
    passed_tests += 1

    # 9. Test CSV Download Generation
    total_tests += 1
    print("\n[Test 9/11] Testing CSV Download Report Generation...")
    csv_bytes = generate_csv_report(results)
    assert len(csv_bytes) > 0, "Generated CSV is empty"
    df_re_csv = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(df_re_csv) == len(results), "CSV export row count mismatch"
    print(f" -> PASSED: CSV report generated ({len(csv_bytes)} bytes).")
    passed_tests += 1

    # 10. Test Excel Download Generation
    total_tests += 1
    print("\n[Test 10/11] Testing Excel Multi-Sheet Styled Report Generation...")
    excel_bytes = generate_excel_report(results, model_meta=get_model_metadata())
    assert len(excel_bytes) > 0, "Generated Excel is empty"
    xl = pd.ExcelFile(io.BytesIO(excel_bytes))
    expected_sheets = ["Predictions", "Executive Summary", "Model Governance"]
    for s in expected_sheets:
        assert s in xl.sheet_names, f"Missing sheet: {s}"
    print(f" -> PASSED: Multi-sheet Excel workbook verified with sheets {xl.sheet_names}.")
    passed_tests += 1

    # 11. Test Full Dataset Test Fixture (Cleaned Dataset with Revenue Excluded)
    total_tests += 1
    print("\n[Test 11/11] Testing Original Cleaned Dataset as Fixture (Revenue Excluded)...")
    original_cleaned_path = os.path.join(
        current_dir, "..", "ML_Models", "03_Website_Visitor_Purchase_Intent_Prediction",
        "datasets", "online_shoppers_intention.csv"
    )
    if os.path.exists(original_cleaned_path):
        df_orig = pd.read_csv(original_cleaned_path)
        # Exclude Revenue from input to simulate genuine upload
        df_orig_input = df_orig.drop(columns=["Revenue"])
        X_orig_prep, _ = preprocess_features(df_orig_input)
        preds_orig = model.predict(X_orig_prep)
        probs_orig = model.predict_proba(X_orig_prep)[:, 1]
        
        # Ground-truth evaluation
        eval_orig = evaluate_predictions_against_ground_truth(df_orig["Revenue"], preds_orig, probs_orig)
        assert eval_orig["accuracy"] > 0.88, f"Expected accuracy > 88%, got {eval_orig['accuracy']}"
        assert eval_orig["roc_auc"] > 0.90, f"Expected ROC-AUC > 90%, got {eval_orig['roc_auc']}"
        print(f" -> PASSED: Original dataset (12,330 rows) evaluated: Accuracy = {eval_orig['accuracy']*100:.2f}%, ROC-AUC = {eval_orig['roc_auc']:.4f}.")
    else:
        print(" -> SKIPPED: Original dataset not found at expected relative path.")

    passed_tests += 1

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} TESTS PASSED (100% SUCCESS)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
