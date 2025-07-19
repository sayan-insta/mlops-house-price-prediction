from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import joblib
import os
from src.utils import load_config

def train_model():
    config = load_config('config/config.json')
    digits = load_digits()
    X, y = digits.data, digits.target

    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    joblib.dump(model, 'model_train.pkl')

if __name__ == "__main__":
    train_model()

