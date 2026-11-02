"""Reporting and Export Module for ShopperSense AI.
Handles CSV export, stylized multi-sheet Excel generation, and executive summary metrics.
"""

from typing import Dict, Any, Optional
import io
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def compute_summary_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level summary KPIs from the predictions dataframe."""
    total_sessions = len(results_df)
    if total_sessions == 0:
        return {
            "total_sessions": 0,
            "predicted_purchases": 0,
            "predicted_non_purchases": 0,
            "purchase_intention_rate": 0.0,
            "avg_purchase_probability": 0.0,
            "high_intent_count": 0,
            "moderate_intent_count": 0,
            "low_intent_count": 0,
            "bounce_risk_count": 0
        }

    purchases = int((results_df['Predicted_Class'] == 1).sum())
    non_purchases = total_sessions - purchases
    purchase_rate = (purchases / total_sessions) * 100
    avg_prob = float(results_df['Purchase_Probability'].mean()) * 100

    tier_counts = results_df['Intent_Tier'].value_counts().to_dict()
    high_intent = tier_counts.get("High Intent (>=75%)", 0)
    moderate_intent = tier_counts.get("Moderate Intent (50-74%)", 0)
    low_intent = tier_counts.get("Low Intent (25-49%)", 0)
    bounce_risk = tier_counts.get("Bounce Risk (<25%)", 0)

    return {
        "total_sessions": total_sessions,
        "predicted_purchases": purchases,
        "predicted_non_purchases": non_purchases,
        "purchase_intention_rate": round(purchase_rate, 2),
        "avg_purchase_probability": round(avg_prob, 2),
        "high_intent_count": high_intent,
        "moderate_intent_count": moderate_intent,
        "low_intent_count": low_intent,
        "bounce_risk_count": bounce_risk
    }


def generate_csv_report(results_df: pd.DataFrame) -> bytes:
    """Generate CSV bytes for download."""
    return results_df.to_csv(index=False).encode('utf-8')


def generate_excel_report(results_df: pd.DataFrame, model_meta: Optional[Dict[str, Any]] = None) -> bytes:
    """Generate a multi-sheet, professionally formatted Excel workbook (.xlsx).
    
    Sheets:
    1. 'Predictions' — Customer session predictions and confidence scores.
    2. 'Executive Summary' — High-level conversion metrics and segmentation.
    3. 'Model Governance' — Model version, parameters, and benchmark metrics.
    """
    output = io.BytesIO()
    metrics = compute_summary_metrics(results_df)

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Predictions
        results_df.to_excel(writer, sheet_name='Predictions', index=False)

        # Sheet 2: Executive Summary
        summary_rows = [
            {"Metric": "Total Customer Sessions Processed", "Value": metrics["total_sessions"]},
            {"Metric": "Predicted Purchases (Class 1)", "Value": metrics["predicted_purchases"]},
            {"Metric": "Predicted Non-Purchases (Class 0)", "Value": metrics["predicted_non_purchases"]},
            {"Metric": "Purchase Intention Rate", "Value": f"{metrics['purchase_intention_rate']}%"},
            {"Metric": "Average Purchase Probability", "Value": f"{metrics['avg_purchase_probability']}%"},
            {"Metric": "High Intent Sessions (>= 75%)", "Value": metrics["high_intent_count"]},
            {"Metric": "Moderate Intent Sessions (50 - 74%)", "Value": metrics["moderate_intent_count"]},
            {"Metric": "Low Intent Sessions (25 - 49%)", "Value": metrics["low_intent_count"]},
            {"Metric": "Bounce Risk Sessions (< 25%)", "Value": metrics["bounce_risk_count"]}
        ]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)

        # Sheet 3: Model Governance
        meta_rows = [
            {"Parameter / Metric": "Application Name", "Details": "ShopperSense AI — Online Shopper Purchase Intention Prediction"},
            {"Parameter / Metric": "Model Architecture", "Details": "RandomForestClassifier (scikit-learn)"},
            {"Parameter / Metric": "Estimators (n_estimators)", "Details": "100"},
            {"Parameter / Metric": "Class Weight", "Details": "balanced"},
            {"Parameter / Metric": "Random State", "Details": "42"},
            {"Parameter / Metric": "Engineered Features Included", "Details": "Total_Pages, Total_Duration, Avg_Duration_Per_Page, Bounce_Exit_Avg"},
            {"Parameter / Metric": "Total Input Features", "Details": "30 aligned features"},
            {"Parameter / Metric": "Documented Test Accuracy", "Details": "89.59% (~89%)"},
            {"Parameter / Metric": "Documented Class 1 Precision", "Details": "0.65"},
            {"Parameter / Metric": "Documented Class 1 Recall", "Details": "0.72"},
            {"Parameter / Metric": "Documented Class 1 F1-Score", "Details": "0.68"},
            {"Parameter / Metric": "Documented ROC-AUC Score", "Details": "0.9257"},
            {"Parameter / Metric": "Scaler Usage", "Details": "Bypassed (Random Forest operates on unscaled features)"}
        ]
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_excel(writer, sheet_name='Model Governance', index=False)

    # Re-open with openpyxl to apply sleek SaaS styling
    output.seek(0)
    wb = openpyxl.load_workbook(output)

    # Styles
    header_fill = PatternFill(start_color="312E81", end_color="312E81", fill_type="solid")  # Deep Indigo
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=10)
    border_thin = Border(
        left=Side(style='thin', color='E2E8F0'),
        right=Side(style='thin', color='E2E8F0'),
        top=Side(style='thin', color='E2E8F0'),
        bottom=Side(style='thin', color='E2E8F0')
    )

    for ws in wb.worksheets:
        ws.views.sheetView[0].showGridLines = True
        for col_idx, col in enumerate(ws.columns, 1):
            max_len = 0
            for row_idx, cell in enumerate(col, 1):
                cell.font = data_font
                cell.border = border_thin
                if row_idx == 1:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                val_str = str(cell.value or '')
                max_len = max(max_len, len(val_str))
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    styled_output = io.BytesIO()
    wb.save(styled_output)
    return styled_output.getvalue()


def generate_summary_markdown(results_df: pd.DataFrame, model_meta: Optional[Dict[str, Any]] = None) -> str:
    """Generate an executive Markdown summary report."""
    metrics = compute_summary_metrics(results_df)
    meta = model_meta or {}
    bench = meta.get("documented_benchmarks", {})

    report = f"""# ShopperSense AI — Executive Prediction Summary Report

**Generated on:** Real-time Analysis  
**Application:** ShopperSense AI — Online Shopper Purchase Intention Prediction  
**Model Architecture:** Random Forest Classifier (`n_estimators=100`, `class_weight='balanced'`, `random_state=42`)

---

## 1. Executive Summary KPIs

- **Total Customer Sessions Processed:** {metrics['total_sessions']:,}
- **Predicted Purchases:** {metrics['predicted_purchases']:,} ({metrics['purchase_intention_rate']}%)
- **Predicted Non-Purchases:** {metrics['predicted_non_purchases']:,} ({100 - metrics['purchase_intention_rate']:.2f}%)
- **Average Purchase Probability:** {metrics['avg_purchase_probability']}%

### Customer Intent Breakdown:
- **High Intent (>= 75% probability):** {metrics['high_intent_count']:,} sessions (Immediate conversion priority)
- **Moderate Intent (50 - 74% probability):** {metrics['moderate_intent_count']:,} sessions (Nudge via cart reminders / discounts)
- **Low Intent (25 - 49% probability):** {metrics['low_intent_count']:,} sessions (Exploratory / window shopping)
- **Bounce Risk (< 25% probability):** {metrics['bounce_risk_count']:,} sessions (Exit intent engagement required)

---

## 2. Model Baseline Governance

The predictions were generated using the validated Random Forest Classifier artifact:
- **Baseline Test Accuracy:** {bench.get('overall_accuracy_str', '89.59%')}
- **Baseline Precision (Class 1):** {bench.get('precision_class_1', '0.65')}
- **Baseline Recall (Class 1):** {bench.get('recall_class_1', '0.72')}
- **Baseline F1-Score (Class 1):** {bench.get('f1_score_class_1', '0.68')}
- **Baseline ROC-AUC:** {bench.get('roc_auc', '0.9257')}

*Note: Model predictions represent statistical probability estimates based on behavioral browsing patterns, not guaranteed purchases.*

---
*Report generated automatically by ShopperSense AI Engine.*
"""
    return report
