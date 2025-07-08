from zenml import step
from typing import Annotated, Tuple
import pandas as pd
import logging
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

@step(enable_cache=False)
def preprocess_data(
    X_train: Annotated[pd.DataFrame, "X_train"],
    X_test: Annotated[pd.DataFrame, "X_test"],
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_preprocessed"],
    Annotated[pd.DataFrame, "X_test_preprocessed"],
]:

    """Preprocess synthetic data."""
    logging.info(" starting processing...")

   
    categorical_cols = ["residence", "job_title", "industry"]
    ordinal_cols = ["education_level"]

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ode = OrdinalEncoder()

    ct = make_column_transformer(
        (ohe, categorical_cols),
        (ode, ordinal_cols),
        remainder="passthrough"
    )
    ct.set_output(transform="pandas")

    logging.info(" transformer done...")

    print("X_train")
    print(X_train.info())
    print("X_test")
    print(X_test.info())

    X_train_transformed = ct.fit_transform(X_train)
    X_test_transformed = ct.transform(X_test)

    logging.info(" processing data complete...")

    print("X_train_transformed")   
    print(X_train_transformed.info()) 

    return X_train_transformed, X_test_transformed

