from sklearn.datasets import load_boston
import pandas as pd

def load_data():
    """
    Load Boston Housing dataset from sklearn and return as DataFrame.
    """
    # Load the dataset
    boston = load_boston()
    # Create DataFrame
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['MEDV'] = boston.target  # MEDV is the target variable (house prices)
    return data

