import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def run_drift_check(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    os.makedirs("reports", exist_ok=True)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    report.save_html("reports/drift_report.html")

    # 🔥 EXTRACT DRIFT RESULT
    result = report.as_dict()

    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    print("[SUCCESS] Drift report generated -> reports/drift_report.html")
    print(f"Drift detected: {drift_detected}")

    return drift_detected