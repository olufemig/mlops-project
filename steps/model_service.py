"""step for deploying model service."""
import bentoml
from zenml import step
from zenml.client import Client
from mlflow.tracking import MlflowClient
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTracker
from typing import Optional

@step
def deploy_bentoml_model(model_name: str, pipeline_name: str, step_name: str) -> Optional[str]:
    """
    Deploy a linear regression model from MLflow using BentoML.

    Returns:
        The Bento model tag or None if deployment fails.
    """
    try:
        zenml_client = Client()
        experiment_tracker = zenml_client.active_stack.experiment_tracker

        if not isinstance(experiment_tracker, MLFlowExperimentTracker):
            raise ValueError("This step requires an MLflow experiment tracker in the active ZenML stack")

        experiment_tracker.configure_mlflow()
        mlflow_client = MlflowClient()

        run_id = experiment_tracker.get_run_id(
            experiment_name=pipeline_name,
            run_name=zenml_client.get_run_name()
        )

        model_uri = f"runs:/{run_id}/{model_name}"

        bento_model = bentoml.mlflow.import_model(
            name=model_name,
            model_uri=model_uri,
            signatures={"predict": {"batchable": True}}
        )

        print(f"Model imported to BentoML with tag: {bento_model.tag}")
        print("You can now define a BentoML service separately.")

        return str(bento_model.tag)

    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        return None
