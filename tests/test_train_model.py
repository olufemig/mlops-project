import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steps.reg_trainer import train_model  # Adjust this path to match your setup
from sklearn.linear_model import LinearRegression


@pytest.fixture
def sample_training_data():
    X_train = pd.DataFrame({
        "feature_1": np.random.rand(50),
        "feature_2": np.random.rand(50),
        "feature_3": np.random.rand(50),
    })
    y_train = pd.Series(np.random.rand(50))
    return X_train, y_train


def test_train_model_fits_and_returns_linear_regression(sample_training_data):
    X_train, y_train = sample_training_data

    with patch("steps.train_model.mlflow.sklearn.log_model") as mock_log_model:
        model = train_model(X_train, y_train)

        # Validate model type
        assert isinstance(model, LinearRegression)

        # Validate model was fitted (check for coef_ attribute)
        assert hasattr(model, "coef_")
        assert model.coef_ is not None

        # Validate MLflow model logging was called
        mock_log_model.assert_called_once()
