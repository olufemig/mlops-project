from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Dict, Tuple, Annotated
import pandas as pd
import numpy as np
import logging
from zenml import pipeline, step


@step (enable_cache=False)
def reg_trainer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,   
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    Annotated[LinearRegression, "trained_model"],
    Annotated[float, "rmse"],
]:
    """Train a sklearn linear regression classifier."""

    model = LinearRegression()
    logging.info("fitting the model...")
    model.fit(X_train, y_train)
    logging.info("running predictions...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info("modelling complete...")

    return model, rmse

