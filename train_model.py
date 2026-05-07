import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate synthetic training data
# Rain tends to occur at high humidity (>70%) and moderate-to-low temperatures
np.random.seed(42)
n = 2000

temperature = np.random.uniform(-10, 45, n)   # Celsius
humidity    = np.random.uniform(10, 100, n)    # Percent

# Rule: rain if humidity > 70 + some noise based on temperature
rain_prob = (humidity / 100) * 0.8 + (1 - (temperature / 45)) * 0.2
noise = np.random.normal(0, 0.1, n)
rain = ((rain_prob + noise) > 0.65).astype(int)

X = np.column_stack([temperature, humidity])
y = rain

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")