## **Salary Prediction Machine Learning Project**

Problem statement

This project builds a machine learning pipeline to predict salaries based on variables such as number of certifications, residence, age, education, and job title. Each run uses auto-generated synthetic data. The pipeline handles data preprocessing, trains a linear regression model using scikit-learn, and evaluates model performance through key metrics. Development was done locally using VS Code, with all data and model artifacts logged and versioned via GitHub.

#### Cloud assets used ( all local stack) 

**Pipelines**: ZenML pipelines for data preprocessing, model training, and evaluation.  
**Monitoring**: Evidently AI dashboards for data drift and model performance monitoring.  
**Experiment Tracking**: MLflow for logging experiments, parameters, metrics, and models.  
**Orchestration**: Apache Airflow DAGs for scheduling and managing pipeline execution ( not completed)  
**CI/CD**: GitHub Actions workflows for automated testing, building, and deployment.  
**API**: FastAPI application serving the model predictions.  
**Deployment**: Model hosted on Hugging Face for inference. ( not completed)

#### Experiment tracking and model register

I used MLflow as both an experiment tracker and a model registry to streamline my machine learning workflow. During experimentation, I log key parameters, metrics, artifacts, and model versions using MLflow Tracking, which helps me compare different runs and identify the most effective models. Once a model meets the desired performance criteria, I register it in the MLflow Model Registry, where I can manage model stages such as staging, production, and archived. This setup provides a structured, auditable process for tracking model lifecycle events, promoting reproducibility and smooth collaboration between development and deployment teams.  
  
 

#### Workflow orchestration

I used ZenML for workflow orchestration by designing modular, reproducible machine learning pipelines that integrate data ingestion, preprocessing, model training, evaluation, and deployment. ZenML allows me to structure these workflows as version-controlled pipelines, leveraging step caching and artifact tracking for efficiency and traceability. Its integration with orchestrators like Kubeflow or Airflow helps scale and automate pipeline runs across environments, ensuring consistent execution from local development to production deployment. I logged the trin and test data as artifacts in ZenML where they could be viewed in the dashboard while the model, evidently report and model metrics were logged to MLFLOW.

Data: Fictitious dataset containing features relevant to IT professionals' salaries (e.g., years of experience, job role, education).

├── data/ # Synthetic test datasets  
├── pipelines/ # ZenML pipeline definitions  
├── airflow/ # Apache Airflow DAGs and configs  
├── tests/ # pytest scripts  
├── steps/ # ZenML pipeline steps  
├── notebooks/ # EDA notebook  
├── mlruns/ # MLFlow data  
├── logged\_datasets/ # GitHub Actions workflows  
├── README.md # Project documentation  
├── run\_pipeline.py # key entry file  
└── requirements.txt # Python dependencies

**Setup and Installation**

```plaintext
Clone the repository:
git clone 
cd
```

```plaintext
Install dependencies:
pip install -r requirements.txt
```

Set up environment variables:Create a .env file and configure necessary keys (e.g., Hugging Face API token, MLflow tracking server).

```plaintext
Initialize ZenML:
zenml init
```

```plaintext
Start Airflow:
airflow webserver -p 8080
airflow scheduler
```

**Run GitHub Actions** :GitHub Actions workflows are configured in .github/workflows/ for CI/CD. i have included code for installing python dependencies, unit testing, linting and uploading the model to hugging spaces for inference

**Running the Project**

```plaintext
Execute the ZenML pipeline:
python run_pipeline.py
```

```plaintext
Track experiments with MLflow:Access the MLflow UI to view experiments:
mlflow ui
```

```plaintext
Deploy the model:Start the FastAPI server:
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The model is deployed to Hugging Face for inference.

**Usage**

Prediction API: Send a POST request to the FastAPI endpoint (e.g., /predict) with input features in JSON format to get salary predictions.  
Monitoring: Check Evidently AI dashboards for data drift and model performance metrics.  
Airflow: Monitor pipeline execution via the Airflow web interface.

**Contributing**

Fork the repository.  
Create a feature branch (git checkout -b feature-name).  
Commit your changes (git commit -m 'Add feature').  
Push to the branch (git push origin feature-name).  
Create a pull request.

License  
This project is licensed under the MIT License. See the LICENSE file for details.
