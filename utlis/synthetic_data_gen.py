import pandas as pd
import numpy as np
df = pd.read_csv('data/hotel_bookings.csv')
original_data = df.copy()

num_synthetic_samples = len(original_data) * 10  # Example: 10x the original data size

large_synthetic_data = pd.DataFrame()

for column in original_data.columns:
    # Randomly sample with replacement for each column
    large_synthetic_data[column] = np.random.choice(original_data[column], size=num_synthetic_samples, replace=True)

# Combine the original data and the generated synthetic data
expanded_data = pd.concat([original_data, large_synthetic_data], axis=0).reset_index(drop=True)

# Check the size of the expanded dataset
print("Original data size:", len(original_data))
print("Expanded data size:", len(expanded_data))
