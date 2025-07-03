from zenml import pipeline, step
from zenml.pipelines import pipeline
from steps.load_data import load_data
from steps.reg_trainer import reg_trainer
import logging

@pipeline (enable_cache=False)

def training_pipeline():
        logging.info("starting pipeline...")
        X_train, X_test, y_train, y_test = load_data()
        reg_trainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

