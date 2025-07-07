from typing import Dict, Tuple, Annotated
import pandas as pd
import numpy as np
import logging
from zenml import pipeline, step



df = pd.read_csv("./data/salary_dataset.csv")
print ( df.head(5))
