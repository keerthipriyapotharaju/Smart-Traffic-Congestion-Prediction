import pandas as pd

# Load dataset
data = pd.read_csv("data/traffic.csv", low_memory=False)

# Convert date_time safely (handles mixed formats)
data['date_time'] = pd.to_datetime(
    data['date_time'],
    errors='coerce',
    dayfirst=True
)

# Drop rows where date_time could not be parsed
data = data.dropna(subset=['date_time'])

# Extract time-based features
data['hour'] = data['date_time'].dt.hour
data['day'] = data['date_time'].dt.day
data['month'] = data['date_time'].dt.month
data['weekday'] = data['date_time'].dt.weekday

# Drop original datetime column
data.drop('date_time', axis=1, inplace=True)

# Fill remaining missing values
data.fillna(0, inplace=True)

# Final dataset check
print("Preprocessed data preview:")
print(data.head())

print("\nFinal dataset shape:", data.shape)
