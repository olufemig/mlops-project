from zenml import step
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow
from zenml.client import Client
import logging

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker="local_mlflow_tracker",enable_cache=False)
def detect_data_report(
    ref_data,  # Reference DataFrame (e.g., train)
    prod_data  # Production DataFrame (e.g., test or live)
) -> None:
    """
    Detects data drift between training and test datasets using Evidently AI and saves the report
    as an HTML string in the ZenML artifact store.

    Args:
        train_data (pd.DataFrame): Reference dataset (training data).
        test_data (pd.DataFrame): Current dataset (test data).

    Returns:
        Annotated[HTMLString, "data_drift_report"]: HTML string of the data drift report.

    Raises:
        ValueError: If input DataFrames are invalid or incompatible.
    """
    # Generate drift report
    logging.info("drift report starting...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=prod_data)

    # Save report as HTML
    report_file = "evidently_drift_report.html"
    report.save_html(report_file)
    # Log to MLflow
    mlflow.log_artifact(report_file)
    logging.info("drift report complete...")