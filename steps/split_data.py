from zenml import step
from sklearn.model_selection import train_test_split
from typing import Annotated, Tuple
import pandas as pd
import logging
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def split_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

    """split synthetic data into training and test."""

    X = df.drop(columns=["salary"])
    logging.info("print X Dataframe...")
    y = df["salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("split completed...")
    return X_train, X_test, y_train, y_test
