import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv("data/traffic.csv", low_memory=False)

# Datetime conversion
data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce', dayfirst=True)
data = data.dropna(subset=['date_time'])

# Feature engineering
data['hour'] = data['date_time'].dt.hour
data['day'] = data['date_time'].dt.day
data['month'] = data['date_time'].dt.month
data['weekday'] = data['date_time'].dt.weekday
data.drop('date_time', axis=1, inplace=True)

# Encode categorical columns
categorical_cols = ['holiday', 'weather_main', 'weather_description']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Fill missing values
data.fillna(0, inplace=True)

# Split
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:100], label="Actual Traffic Volume")
plt.plot(y_pred[:100], label="Predicted Traffic Volume")
plt.xlabel("Sample Index")
plt.ylabel("Traffic Volume")
plt.title("Actual vs Predicted Traffic Volume")
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig("traffic_prediction_result.png")
plt.show()
