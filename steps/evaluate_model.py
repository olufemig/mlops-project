from zenml import pipeline, step
from typing import Annotated, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

@step
def evaluate_model(
    model: LinearRegression,
    X_test: Annotated[pd.DataFrame, "X_test_preprocessed"],
    y_test: Annotated[pd.Series, "y_test"],
) -> float:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return mse