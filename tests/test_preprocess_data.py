import pytest
import pandas as pd
import sys
import os


from steps.preprocess_data import preprocess_data 


@pytest.fixture
def sample_dataframes():
    data = {
        "residence": ["San Francisco", "Austin", "London"],
        "job_title": ["Data Scientist", "Analyst", "Software Engineer"],
        "industry": ["Tech", "Finance", "Education"],
        "education_level": ["Masters", "PhD", "Bachelors"],
        "experience_years": [5, 10, 2],
        "certifications": [1, 3, 0],
    }

    df_train = pd.DataFrame(data)
    df_test = pd.DataFrame(data)

    return df_train, df_test


def test_preprocess_data_output_shape_and_columns(sample_dataframes):
    df_train, df_test = sample_dataframes

    X_train_transformed, X_test_transformed = preprocess_data(df_train, df_test)

    # Basic shape checks
    assert X_train_transformed.shape[0] == df_train.shape[0]
    assert X_test_transformed.shape[0] == df_test.shape[0]

    # Check if no null values exist after transformation
    assert not X_train_transformed.isnull().any().any()
    assert not X_test_transformed.isnull().any().any()

    # Check if categorical and ordinal features have been encoded
    encoded_columns = set(X_train_transformed.columns) - set(df_train.columns)

    assert any("residence" in col for col in encoded_columns) or \
           any("job_title" in col for col in encoded_columns)

    # Verify education_level is ordinal-encoded (converted to float)
    education_col = [col for col in X_train_transformed.columns if "education_level" in col]
    assert len(education_col) == 1
    assert pd.api.types.is_numeric_dtype(X_train_transformed[education_col[0]])


def test_preprocess_data_consistency(sample_dataframes):
    df_train, df_test = sample_dataframes

    X_train_transformed, X_test_transformed = preprocess_data(df_train, df_test)

    # Ensure both outputs have same columns
    assert list(X_train_transformed.columns) == list(X_test_transformed.columns)
