from zenml import pipeline, step
from typing import Annotated, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

@step
def evaluate_model(
    model: LinearRegression,
    X_test: Annotated[pd.DataFrame, "X_test_preprocessed"],
    y_test: Annotated[pd.Series, "y_test"],
) -> float:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    print(f"Mean Squared Error: {mse}")
    return mse