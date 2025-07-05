from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    """
    Load California Housing dataset from sklearn and return as DataFrame.
    """
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame
    return data

def split_data(data):
    """
    Split the data into features and target.
    """
    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']
    return X, y

