import pandas as pd
import random
import logging
#from faker import Faker
from zenml import step

@step(enable_cache=False)
def generate_salary_data(n=500)-> pd.DataFrame:


    #fake = Faker()
    data = []

    logging.info("generating synthetic data...")

    for _ in range(n):
        experience = random.randint(0, 20)
        education = random.choice(["Bachelors", "Masters", "PhD"])
        job = random.choice(["Data Scientist", "Software Engineer", "Analyst", "Account Executive"])
        industry = random.choice(["Tech", "Finance", "Healthcare", "Education"])
        residence = random.choice(["San Francisco", "Austin", "New York", "London"])
        certs = random.randint(0, 5)

        # base salary
        base = 30000 + experience * 2500 + certs * 1500

        # education bonus
        if education == "Masters":
            base += 10000
        elif education == "PhD":
            base += 20000

        # job multiplier
        job_multipliers = {
            "Data Scientist": 1.2,
            "Software Engineer": 1.15,
            "Account Executive": 1.10,
            "Analyst": 1.0
        }

        # location adjustment
        location_adjustment = {
            "San Francisco": 1.3,
            "New York": 1.2,
            "Austin": 1.1,
            "London": 1.15
        }

        salary = base * job_multipliers[job] * location_adjustment[residence]

        data.append({
            "experience_years": experience,
            "education_level": education,
            "job_title": job,
            "industry": industry,
            "residence": residence,
            "certifications": certs,
            "salary": round(salary, 2)
        })
  

    df= pd.DataFrame(data)
    return df
    print(df.info())
    logging.info(" synthetic data generated...")