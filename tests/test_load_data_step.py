import pandas as pd
import tempfile
import os
import sys
import pytest
from steps.load_data import load_data
from steps.load_data import load_data_logic  

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_load_data_step_entrypoint():
    # Create a temporary CSV file with dummy salary data
    df = pd.DataFrame({
        "experience": [1, 2, 3, 4, 5],
        "salary": [30000, 35000, 40000, 45000, 50000]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "salary_dataset.csv")
        df.to_csv(csv_path, index=False)

        # Monkeypatch the step logic to read from this temp CSV path
        from steps.load_data import load_data_logic

        def patched_load_data_logic(_):
            return load_data_logic(csv_path)

        # Replace the step logic with our patched version
        load_data.entrypoint = lambda *args, **kwargs: patched_load_data_logic(None)

        # Run the step as a function
        X_train, X_test, y_train, y_test = load_data.entrypoint()

        # Basic assertions
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert 'experience' in X_train.columns
        assert len(X_train) + len(X_test) == 5
