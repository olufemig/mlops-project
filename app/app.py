import gradio as gr
import pickle
import pandas as pd

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the prediction function
def predict_salary(experience, education, job, industry, residence, certs):
    # Create a single-row DataFrame to match training schema
    input_df = pd.DataFrame([{
        "experience": int(experience),
        "education": education,
        "job": job,
        "industry": industry,
        "residence": residence,
        "certs": int(certs)
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    return f"${prediction:,.2f}"

# Define Gradio interface
interface = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Slider(0, 30, step=1, label="Years of Experience"),
        gr.Dropdown(["Bachelors", "Masters", "PhD"], label="Education Level"),
        gr.Dropdown(["Data Scientist", "Software Engineer", "Analyst", "Account Executive"], label="Job Title"),
        gr.Dropdown(["Tech", "Finance", "Healthcare", "Education"], label="Industry"),
        gr.Dropdown(["San Francisco", "Austin", "New York", "London"], label="City of Residence"),
        gr.Slider(0, 10, step=1, label="Number of Certifications")
    ],
    outputs="text",
    title="Salary Prediction App",
    description="Enter job information to estimate salary using a trained ML model.",
)

interface.launch()
