kfrom sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    """
    Load California Housing dataset from sklearn and return as DataFrame.
    """
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame
    return data

def split_data(data, target_column):
    """
    Split the data into training and testing sets.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

