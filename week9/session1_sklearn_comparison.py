import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- Load data ---
data = fetch_california_housing()
X, y = data.data, data.target

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- sklearn model ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print("=== sklearn Linear Regression ===")
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test  R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"Test  MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"\nWeights: {model.coef_}")
print(f"Bias:    {model.intercept_:.4f}")
