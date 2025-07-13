import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from typing import Annotated
import pandas as pd
from zenml import step
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

@step (experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(
    X_train: Annotated[pd.DataFrame, "X_train_preprocessed"],
    y_train: Annotated[pd.Series, "y_train"],
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "sklearn_model")
    return model



