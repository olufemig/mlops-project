from pipelines.pipeline import training_pipeline
from steps.load_data import load_data
from steps.reg_trainer import reg_trainer


if __name__ == "__main__":
 training_pipeline()