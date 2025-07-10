from zenml import pipeline, step
from typing import Annotated, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model(
    model: LinearRegression,
    X_test: Annotated[pd.DataFrame, "X_test_preprocessed"],
    y_test: Annotated[pd.Series, "y_test"],
) -> float:
    

    print("📍 Tracking URI:", mlflow.get_tracking_uri())
    print("🔁 Active run:", mlflow.active_run())

    mlflow.log_metric("debug_metric", 123)
    
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    print(f"Mean Squared Error: {mse}")

    return mse
