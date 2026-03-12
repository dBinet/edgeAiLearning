import numpy as np
import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# To improve accuracy a few things were attempted
# 1 increase the number of epochs and set a check when it starts to converge (no difference betwee current and previous epoch). This helped getting the most of our model
# 2 Reduce the number of features. This one didn't worked in the end and decrease the accuracy. Used correlation to make a decision on which to drop but those features
# ended up being useful 


# --- sigmoid function ---
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

# --- binary cross-entropy loss function ---
def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- train function ---
def train(X, y, learning_rate=0.1, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
        # 1. forward pass — compute predictions
        y_pred = sigmoid_function(X @ w + b)
        # 2. compute loss
        loss = compute_loss(y, y_pred)
        # 3. compute gradients dw and db
        dw = (1 / n_samples) * X.T @ (y_pred - y)
        db = np.mean(y_pred - y)
        # 4. update w and b
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 5. print loss every 100 epochs
        if (epoch % 100 == 0):
            print(f"Loss: {loss}")
        if epoch > 0 and abs(prev_loss - loss) < 1e-6:
            print(f"Converged at epoch {epoch}")
            break
        prev_loss = loss

    return w, b

def predict(X, w, b):
    y_pred = sigmoid_function(X @ w + b)
    return (y_pred >= 0.5).astype(int)  # convert probabilities to 0 or 1



# Load data
data = load_breast_cancer()
X, y = data.data, data.target

cols_to_drop = [
    'mean perimeter', 'mean area',
    'worst perimeter', 'worst area',
    'perimeter error', 'area error',
    'mean concave points'
]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = pd.DataFrame(X_train, columns=data.feature_names).drop(columns=cols_to_drop)
df_test = pd.DataFrame(X_test, columns=data.feature_names).drop(columns=cols_to_drop)

X_train_reduced = df_train.values
X_test_reduced = df_test.values

# Normalize
scaler = StandardScaler()
X_train_reduced = scaler.fit_transform(X_train_reduced)
X_test_reduced = scaler.transform(X_test_reduced)

w, b = train(X_train_reduced, y_train, 0.1, 10000)

y_pred_test = predict(X_test_reduced, w, b)
accuracy = np.mean(y_pred_test == y_test)
print(f"From-scratch accuracy: {accuracy:.4f}")

# From scracth algorithm before a bit better but it's because there is no safety that would avoid overfitting the data.
# This could become a problem on noisier data where overfitting is more likely to happen.
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
print(f"sklearn accuracy: {clf.score(X_test, y_test):.4f}")

