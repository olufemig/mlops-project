from zenml import pipeline, Model
from zenml.pipelines import pipeline
from steps.split_data import split_data
from steps.reg_trainer import train_model
from steps.extract_data import extract_data
from steps.detect_data_drift import detect_data_drift
from steps.preprocess_data import preprocess_data
from steps.generate_synthetic_data import generate_salary_data
from steps.evaluate_model import evaluate_model
import logging
from pathlib import Path

@pipeline (enable_cache=False)

def training_pipeline():
        logging.info("starting data pipeline...")
        df = generate_salary_data()    
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_prep, X_test_prep = preprocess_data(X_train, X_test)
        drift_report_path = detect_data_drift(X_train_prep, X_test_prep)
        model = train_model(X_train_prep, y_train)
        evaluate_model(model, X_test_prep, y_test)

