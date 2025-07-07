from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Annotated
import pandas as pd
import numpy as np
import logging
from zenml import pipeline, step

@step(enable_cache=False)
def load_data(df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = df['salary']
    X = df[['experience']]
    logging.info("Splitting into train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test

