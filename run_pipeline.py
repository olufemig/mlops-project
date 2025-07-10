from pipelines.pipeline import training_pipeline
from steps.split_data import split_data
from steps.reg_trainer import train_model
from steps.preprocess_data import preprocess_data
from steps.generate_synthetic_data import generate_salary_data
from steps.evaluate_model import evaluate_model


if __name__ == "__main__":
 training_pipeline()