import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.detect_data_drift import detect_data_report


@pytest.fixture
def sample_dataframes():
    df_ref = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8]
    })

    df_prod = pd.DataFrame({
        "feature1": [1, 2, 3, 5],
        "feature2": [5, 6, 7, 9]
    })

    return df_ref, df_prod


def test_detect_data_report_runs_and_logs_artifact(sample_dataframes):
    ref_data, prod_data = sample_dataframes

    with patch("steps.detect_data_drift.Report") as mock_report_cls, \
         patch("steps.detect_data_drift.mlflow.log_artifact") as mock_log_artifact:

        # Mock the report instance
        mock_report_instance = MagicMock()
        mock_report_cls.return_value = mock_report_instance

        detect_data_report(ref_data, prod_data)

        # Assert Report was instantiated with metrics
        mock_report_cls.assert_called_once()
        
        # Assert report.run was called with correct data
        mock_report_instance.run.assert_called_once_with(
            reference_data=ref_data,
            current_data=prod_data
        )

        # Assert report.save_html was called
        mock_report_instance.save_html.assert_called_once_with("evidently_drift_report.html")

        # Assert MLflow artifact logging happened
        mock_log_artifact.assert_called_once_with("evidently_drift_report.html")
