from zenml import pipeline, Model
from zenml.pipelines import pipeline
from steps.load_data import load_data
from steps.reg_trainer import reg_trainer
from steps.extract_data import extract_data
from steps.generate_synthetic_data import generate_salary_data
import logging
from pathlib import Path

@pipeline (enable_cache=False)

def training_pipeline():
        logging.info("starting pipeline...")
        synthetic_data = generate_salary_data
        X_train, X_test, y_train, y_test = load_data(synthetic_data)
        model, rmse = reg_trainer(X_train, X_test, y_train, y_test)

