import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Load data ---
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {data.feature_names}")
print(f"Classes: {data.target_names}")
print(f"Class distribution: {np.bincount(y)}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Train ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

print(f"\nTrain Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test  Accuracy: {accuracy_score(y_test,  y_pred_test):.4f}")

print(f"\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test))
print("Rows = Actual | Columns = Predicted")
print(f"[Malignant, Benign]")