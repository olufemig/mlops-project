from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Annotated
import pandas as pd
import numpy as np
import logging
from zenml import pipeline, step

@step (enable_cache=False)
def load_data() -> Tuple[
    # Notice we use a Tuple and Annotated to return 
    # multiple named outputs
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    logging.info("Loading data...")
    df = pd.read_csv("data/salary_dataset.csv")
    y = df['salary']
    X = df[['experience']]
    logging.info("Splitting into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    logging.info("Split complete...")
    return X_train, X_test, y_train, y_test