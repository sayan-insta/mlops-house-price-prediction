import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data, split_data

def main():
    # Load the dataset
    data = load_data()

    # Set correct target column name for California housing dataset
    target_column = 'MedHouseVal'

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

if __name__ == "__main__":
    main()

