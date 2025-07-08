from zenml import step
import evidently
from zenml.logger import get_logger
from typing import Annotated, Tuple
from evidently import Report
from evidently.metrics import DriftedColumnsCount
import pandas as pd
import logging


@step
def detect_data_drift(
    reference_data: Annotated[pd.DataFrame, "X_train_prep"],
    comparison_data: Annotated[pd.DataFrame, "X_test_prep"],
) -> Annotated[str, "drift_report_path"]:
    """Detects data drift between training and test datasets using Evidently."""

    report = Report(metrics=[DriftedColumnsCount()])
    report.run(reference_data=reference_data, current_data=comparison_data)

    # Save report as HTML
    report_path = "data_drift_report.html"
    #report.save(path=report_path, format="html")

    logging.info(f"Data drift report saved to {report_path}")
    return report_path
