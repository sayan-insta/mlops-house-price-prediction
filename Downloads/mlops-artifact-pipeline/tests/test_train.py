import json
import pytest
from src.utils import load_config
from src.train import train_model
from sklearn.linear_model import LogisticRegression
import joblib
import os

def test_config_load():
    config = load_config("config/config.json")
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_creation():
    train_model()
    assert os.path.exists("model_train.pkl")
    model = joblib.load("model_train.pkl")
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_model_accuracy():
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    model = joblib.load("model_train.pkl")
    acc = model.score(X, y)
    assert acc > 0.8  # baseline threshold

