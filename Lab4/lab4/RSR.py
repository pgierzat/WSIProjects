import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionModel:
    def __init__(self, regularization=None, lambda_=0, learning_rate=None):
        self.regularization = regularization
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0

    def train(self, X, y, learning_rate=0.01, n_epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)

        for _ in range(n_epochs):
            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y

            gradient_w = (2 / n_samples) * np.dot(X.T, errors)
            gradient_b = (2 / n_samples) * np.sum(errors)

            if self.regularization == "L1":
                gradient_w += self.lambda_ * np.sign(self.weights)
            elif self.regularization == "L2":
                gradient_w += 2 * self.lambda_ * self.weights

            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2


data = pd.read_csv("fish_species.csv")

X = data.drop(columns=["Weight"])
y = data["Weight"]

X = pd.get_dummies(X, columns=["Species"])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

best_l1_model = None
best_l2_model = None
best_l1_score = float("inf")
best_l2_score = float("inf")

results = []
lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5]
learning_rates = [0.001, 0.01, 0.1]

for learning_rate in learning_rates:
    for lambda_ in lambdas:
        model_l1 = LinearRegressionModel(regularization="L1", lambda_=lambda_, learning_rate=learning_rate)
        model_l1.train(X_train, y_train, learning_rate=learning_rate, n_epochs=1000)
        mse, r2 = model_l1.evaluate(X_val, y_val)
        results.append({
            "Regularization": "L1",
            "Lambda": lambda_,
            "Learning Rate": learning_rate,
            "MSE": mse,
            "R2": r2
        })
        if mse < best_l1_score:
            best_l1_score = mse
            best_l1_model = model_l1

        model_l2 = LinearRegressionModel(regularization="L2", lambda_=lambda_, learning_rate=learning_rate)
        model_l2.train(X_train, y_train, learning_rate=learning_rate, n_epochs=1000)
        mse, r2 = model_l2.evaluate(X_val, y_val)
        results.append({
            "Regularization": "L2",
            "Lambda": lambda_,
            "Learning Rate": learning_rate,
            "MSE": mse,
            "R2": r2
        })
        if mse < best_l2_score:
            best_l2_score = mse
            best_l2_model = model_l2

results_df = pd.DataFrame(results)

print("Best L1 Model Weights:", best_l1_model.weights)
print("Best L2 Model Weights:", best_l2_model.weights)
print("Best L1 Model Parameters:", {"Lambda": best_l1_model.lambda_, "Learning Rate": best_l1_model.learning_rate})
print("Best L2 Model Parameters:", {"Lambda": best_l2_model.lambda_, "Learning Rate": best_l2_model.learning_rate})

l1_test_mse, l1_test_r2 = best_l1_model.evaluate(X_test, y_test)
l2_test_mse, l2_test_r2 = best_l2_model.evaluate(X_test, y_test)

print(f"L1 Test MSE: {l1_test_mse}, R2: {l1_test_r2}")
print(f"L2 Test MSE: {l2_test_mse}, R2: {l2_test_r2}")


l1_val_mse, l1_val_r2 = best_l1_model.evaluate(X_val, y_val)
print(f"Best L1 Model on Validation Set: {l1_val_mse} {l1_val_r2}")

l2_val_mse, l2_val_r2 = best_l2_model.evaluate(X_val, y_val)
print(f"Best L2 Model on Validation Set: {l2_val_mse} {l2_val_r2}")


l1_predictions = best_l1_model.predict(X_test)
l2_predictions = best_l2_model.predict(X_test)

plt.figure(figsize=(12, 8))
plt.scatter(range(len(y_test)), y_test, color='gray', alpha=0.6, label='Rzeczywiste wartości')

plt.plot(range(len(l1_predictions)), l1_predictions, color='blue', label='Best L1 Model Predictions')
plt.plot(range(len(l2_predictions)), l2_predictions, color='red', label='Best L2 Model Predictions')

plt.xlabel('Index')
plt.ylabel('Weight')
plt.title('Liniowa Regresja: Przewidywane vs Rzeczywiste Wartości')
plt.legend()
plt.grid()
plt.savefig("linear.png")


l1_predictions = best_l1_model.predict(X_test)
l2_predictions = best_l2_model.predict(X_test)

plt.figure(figsize=(12, 6))

l1_diff = y_test - l1_predictions
plt.scatter(range(len(l1_diff)), l1_diff, color='blue', alpha=0.6, label='L1')
plt.axhline(0, color='blue', linestyle='--')


l2_diff = y_test - l2_predictions
plt.scatter(range(len(l2_diff)), l2_diff, color='red', alpha=0.6, label='L2')
plt.axhline(0, color='red', linestyle='--')

plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Różnica między rzeczywistą a przewidywaną wagą (L1 i L2)')
plt.legend()
plt.tight_layout()
plt.savefig("diff.png")
