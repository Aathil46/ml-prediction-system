"""ShopperSense AI utilities package."""
from .validation import validate_uploaded_file, validate_input_schema, check_data_quality, REQUIRED_COLUMNS
from .preprocessing import preprocess_features, EXPECTED_MODEL_FEATURES
from .prediction import load_rf_model, predict_purchase_intent, get_model_metadata, evaluate_predictions_against_ground_truth
from .reporting import generate_csv_report, generate_excel_report, generate_summary_markdown, compute_summary_metrics

__all__ = [
    "validate_uploaded_file",
    "validate_input_schema",
    "check_data_quality",
    "REQUIRED_COLUMNS",
    "preprocess_features",
    "EXPECTED_MODEL_FEATURES",
    "load_rf_model",
    "predict_purchase_intent",
    "get_model_metadata",
    "evaluate_predictions_against_ground_truth",
    "generate_csv_report",
    "generate_excel_report",
    "generate_summary_markdown",
    "compute_summary_metrics"
]
