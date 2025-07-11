import pandas as pd
import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.generate_synthetic_data import generate_salary_data


@pytest.fixture
def salary_data_sample():
    return generate_salary_data(n=100)


def test_generate_salary_data_shape_and_columns(salary_data_sample):
    df = salary_data_sample

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100

    expected_columns = {
        "experience_years",
        "education_level",
        "job_title",
        "industry",
        "residence",
        "certifications",
        "salary"
    }
    assert set(df.columns) == expected_columns


def test_generate_salary_data_value_ranges(salary_data_sample):
    df = salary_data_sample

    assert df['experience_years'].between(0, 20).all()
    assert df['certifications'].between(0, 5).all()
    assert (df['salary'] > 0).all()

    valid_education = {"Bachelors", "Masters", "PhD"}
    valid_jobs = {"Data Scientist", "Software Engineer", "Analyst", "Account Executive"}
    valid_industries = {"Tech", "Finance", "Healthcare", "Education"}
    valid_residences = {"San Francisco", "Austin", "New York", "London"}

    assert set(df['education_level'].unique()).issubset(valid_education)
    assert set(df['job_title'].unique()).issubset(valid_jobs)
    assert set(df['industry'].unique()).issubset(valid_industries)
    assert set(df['residence'].unique()).issubset(valid_residences)
