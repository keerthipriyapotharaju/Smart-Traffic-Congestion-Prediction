import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load preprocessed data
data = pd.read_csv("data/traffic.csv", low_memory=False)

# Convert date_time safely
data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce', dayfirst=True)
data = data.dropna(subset=['date_time'])

# Feature engineering
data['hour'] = data['date_time'].dt.hour
data['day'] = data['date_time'].dt.day
data['month'] = data['date_time'].dt.month
data['weekday'] = data['date_time'].dt.weekday
data.drop('date_time', axis=1, inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = ['holiday', 'weather_main', 'weather_description']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Handle missing values
data.fillna(0, inplace=True)

# Split features and target
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "traffic_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model training completed and saved successfully.")
