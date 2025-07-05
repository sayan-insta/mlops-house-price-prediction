from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE = {mse:.2f}, R² = {r2:.2f}")

def run_tuned_models():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge": {
            "model": Ridge(),
  "params": {
                "alpha": [0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
                "solver": ["auto", "svd", "cholesky"]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": [2, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [5, 10],
                "min_samples_split": [2, 5]
            }
        }
    }
  for name, config in models.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        evaluate_model(name, best_model, X_test, y_test)

if __name__ == "__main__":
    run_tuned_models()
