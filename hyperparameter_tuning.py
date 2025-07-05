from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def hyperparameter_tuning():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "DecisionTree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10]
            }
        }
    }

    for name, config in models.items():
        grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{name} Best Params: {grid.best_params_}")
        print(f"{name} -> MSE: {mse:.4f}, R²: {r2:.4f}\n")

if __name__ == "__main__":
    hyperparameter_tuning()

