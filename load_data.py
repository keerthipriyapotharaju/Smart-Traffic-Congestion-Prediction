import pandas as pd

# Load dataset
data = pd.read_csv("data/traffic.csv", low_memory=False)

# Display first 5 rows
print(data.head())

# Dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())
