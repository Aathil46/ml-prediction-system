"""Prediction and Model Inference Module for ShopperSense AI.
Handles loading the verified Random Forest model, generating inference & probabilities,
and computing ground-truth evaluation metrics if labels are present.
"""

from typing import Tuple, Dict, Any, Optional
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Relative default path to the model file
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "random_forest_model.pkl")


def load_rf_model(model_path: Optional[str] = None):
    """Load and validate the pretrained Random Forest Classifier.
    
    Args:
        model_path: Optional explicit file path to the model pickle.
        
    Returns:
        Loaded scikit-learn RandomForestClassifier instance.
        
    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If loaded model does not meet the 30-feature schema requirement.
    """
    path = model_path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model file: {str(e)}")

    # Validate feature count
    n_features = getattr(model, 'n_features_in_', None)
    if n_features != 30:
        raise ValueError(f"Loaded model expects {n_features} features, but schema requires 30 features.")

    return model


def predict_purchase_intent(model, X_preprocessed: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Generate purchase intention predictions, probabilities, and intent tiers.
    
    Args:
        model: Loaded Random Forest Classifier.
        X_preprocessed: Preprocessed [N x 30] DataFrame.
        raw_df: Optional original DataFrame to preserve user input fields.
        
    Returns:
        DataFrame containing Session_ID, predictions, probabilities, tiers, and features.
    """
    # 1. Class predictions (0 = No Purchase, 1 = Purchase)
    predictions = model.predict(X_preprocessed)

    # 2. Probability predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_preprocessed)
        prob_no_purchase = probabilities[:, 0]
        prob_purchase = probabilities[:, 1]
    else:
        prob_purchase = predictions.astype(float)
        prob_no_purchase = 1.0 - prob_purchase

    # 3. Build results table
    n_rows = len(predictions)
    session_ids = [f"SESS-{i+1:04d}" for i in range(n_rows)]

    # Intent tiers based on purchase probability
    def assign_tier(p: float) -> str:
        if p >= 0.75:
            return "High Intent (>=75%)"
        elif p >= 0.50:
            return "Moderate Intent (50-74%)"
        elif p >= 0.25:
            return "Low Intent (25-49%)"
        else:
            return "Bounce Risk (<25%)"

    tiers = [assign_tier(p) for p in prob_purchase]
    labels = ["Purchase Intention" if p == 1 else "No Purchase Intention" for p in predictions]

    results = pd.DataFrame({
        "Session_ID": session_ids,
        "Predicted_Class": predictions,
        "Prediction_Label": labels,
        "Purchase_Probability": np.round(prob_purchase, 4),
        "No_Purchase_Probability": np.round(prob_no_purchase, 4),
        "Purchase_Probability_Pct": [f"{p*100:.1f}%" for p in prob_purchase],
        "Intent_Tier": tiers
    })

    # Append original customer/session columns if provided
    if raw_df is not None:
        clean_raw = raw_df.copy().reset_index(drop=True)
        # Avoid column name collision
        for col in clean_raw.columns:
            if col not in results.columns:
                results[col] = clean_raw[col]

    return results


def get_model_metadata() -> Dict[str, Any]:
    """Retrieve verified model architecture, parameters, and baseline training/test benchmarks."""
    return {
        "model_name": "Random Forest Classifier",
        "algorithm": "RandomForestClassifier",
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 42,
        "features_count": 30,
        "target_variable": "Revenue (0 = No Purchase, 1 = Purchase)",
        "documented_benchmarks": {
            "overall_accuracy": 0.8959,
            "overall_accuracy_str": "89.59%",
            "precision_class_1": 0.65,
            "recall_class_1": 0.72,
            "f1_score_class_1": 0.68,
            "roc_auc": 0.9257
        },
        "scaler_status": "Bypassed (Not Required for Random Forest; scaler.pkl was designed for Logistic Regression)"
    }


def evaluate_predictions_against_ground_truth(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Any]:
    """Evaluate predictions against ground-truth labels if 'Revenue' is supplied in the dataset.
    
    Args:
        y_true: Actual binary labels.
        y_pred: Predicted binary classes.
        y_proba: Predicted purchase probabilities.
        
    Returns:
        Evaluation metrics dictionary.
    """
    # Clean boolean or string labels to 0/1 integers
    y_t = pd.Series(y_true).map({
        True: 1, False: 0,
        'True': 1, 'False': 0,
        'true': 1, 'false': 0,
        1: 1, 0: 0, '1': 1, '0': 0
    }).fillna(0).astype(int).values

    acc = accuracy_score(y_t, y_pred)
    prec = precision_score(y_t, y_pred, zero_division=0)
    rec = recall_score(y_t, y_pred, zero_division=0)
    f1 = f1_score(y_t, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_t, y_proba)
    except Exception:
        auc = None

    cm = confusion_matrix(y_t, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc) if auc is not None else "N/A",
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "total_evaluated_samples": len(y_t)
    }
