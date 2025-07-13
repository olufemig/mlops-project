from zenml import pipeline
#from zenml.pipelines import pipeline
from steps.split_data import split_data
from steps.reg_trainer import train_model
from steps.preprocess_data import preprocess_data
from steps.generate_synthetic_data import generate_salary_data
from steps.evaluate_model import evaluate_model
from steps.detect_data_drift import detect_data_report
import logging

@pipeline (enable_cache=False)

def training_pipeline():
        logging.info("starting ZenML pipeline...")
        df = generate_salary_data()    
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_prep, X_test_prep = preprocess_data(X_train, X_test)
        model = train_model(X_train_prep, y_train)
        evaluate_model(model, X_test_prep, y_test)
        detect_data_report(X_train_prep, X_test_prep)

