import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from unittest.mock import patch
import sys
import os


from steps.evaluate_model import evaluate_model  # Adjust path as needed


@pytest.fixture
def trained_model_and_test_data():
    # Create simple linear data
    X_train = pd.DataFrame({"feature": np.arange(10).reshape(-1)})
    y_train = pd.Series(np.arange(10))

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = X_train.copy()
    y_test = y_train.copy()

    return model, X_test, y_test


def test_evaluate_model_outputs_and_metrics(trained_model_and_test_data):
    model, X_test, y_test = trained_model_and_test_data

    with patch("steps.evaluate_model.mlflow.log_metric") as mock_log_metric:
        mse = evaluate_model(model, X_test, y_test)

        # Check type and value of MSE output
        assert isinstance(mse, float)
        assert mse >= 0.0

        # Check mlflow.log_metric called with expected metric keys
        metric_keys = {call.args[0] for call in mock_log_metric.call_args_list}
        expected_keys = {"debug_metric", "mae", "mse", "rmse", "r2"}

        assert expected_keys.issubset(metric_keys)
