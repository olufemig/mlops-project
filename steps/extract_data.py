
from typing import Dict, Tuple, Annotated
import pandas as pd
import numpy as np
import logging
from zenml import pipeline, step


@step(enable_cache=False)
def extract_data() -> pd.DataFrame:

    return pd.read_csv("./data/salary_dataset.csv")

    