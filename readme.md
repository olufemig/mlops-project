**Salary Prediction for IT Professionals**  
Problem statement  
This project is a machine learning experiment designed to predict salaries for IT professionals using fictitious data. It leverages a modern MLOps stack to build, track, deploy, and monitor a machine learning model.

Cloud asects used  
  
Experiment tracking and model registr

Workflow orchestration  
  
Model deployment  
  
Reproducibility  
  
**Technologies Used**

Scikit-learn: Core library for building and training the machine learning model.  
ZenML: For creating and managing ML pipelines.  
Evidently AI: For data observation and monitoring model performance.  
MLflow: For experiment tracking and management.  
Apache Airflow: For orchestrating workflows and scheduling tasks.  
GitHub Actions: For continuous integration and continuous deployment (CI/CD).  
FastAPI: For deploying the model as an API.  
Hugging Face: For hosting the deployed model.

**Project Structure**

Data: Fictitious dataset containing features relevant to IT professionals' salaries (e.g., years of experience, job role, education).  
  
**Pipelines**: ZenML pipelines for data preprocessing, model training, and evaluation.  
**Monitoring**: Evidently AI dashboards for data drift and model performance monitoring.  
**Experiment Tracking**: MLflow for logging experiments, parameters, metrics, and models.  
**Orchestration**: Apache Airflow DAGs for scheduling and managing pipeline execution.  
**CI/CD**: GitHub Actions workflows for automated testing, building, and deployment.  
**API**: FastAPI application serving the model predictions.  
**Deployment**: Model hosted on Hugging Face for inference.

├── data/ # Synthetic datasets  
├── pipelines/ # ZenML pipeline definitions  
├── airflow/ # Apache Airflow DAGs and configs  
├── tests/ # FastAPI app and deployment scripts  
├── pipelines/ # MLflow experiment tracking data  
├── steps/ # GitHub Actions workflows  
├── notebooks/ # GitHub Actions workflows  
├── mlruns/ # GitHub Actions workflows  
├── logged\_datasets/ # GitHub Actions workflows  
├── README.md # Project documentation  
├── run\_pipeline.py # Project documentation  
└── requirements.txt # Python dependencies

**Setup and Installation**

**Clone the repository:**  
git clone   
cd

**Install dependencies:**  
pip install -r requirements.txt

Set up environment variables:Create a .env file and configure necessary keys (e.g., Hugging Face API token, MLflow tracking server).

**Initialize ZenML:**  
zenml init

**Start Airflow:**  
airflow webserver -p 8080  
airflow scheduler

**Run GitHub Actions** :Ensure GitHub Actions workflows are configured in .github/workflows/ for CI/CD.

**Running the Project**

Execute the ZenML pipeline:  
python run\_pipeline.py

**Monitor with Evidently AI:** Generate reports for data drift and model performance:  
python generate\_monitoring\_report.py

Track experiments with MLflow:Access the MLflow UI to view experiments:  
mlflow ui

Deploy the model:Start the FastAPI server:  
uvicorn app.main:app --host 0.0.0.0 --port 8000

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