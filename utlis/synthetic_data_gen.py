import pandas as pd
import numpy as np

df = pd.read_csv('data/cleaned_data.csv')
original_data = df.copy()

num_synthetic_samples = 100

def generate_synthetic_data():
    synthetic_data = pd.DataFrame()
    for column in original_data.columns:
        synthetic_data[column] = np.random.choice(original_data[column], size=num_synthetic_samples, replace=True)
    return synthetic_data

synthetic_data = generate_synthetic_data()
print(synthetic_data.head())
