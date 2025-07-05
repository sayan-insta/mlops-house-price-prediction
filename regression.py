from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def run_regression_models():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        mse, r2 = evaluate_model(model, X_test, y_test)
        results[name] = {"MSE": mse, "R2": r2}

    return results

if __name__ == "__main__":
    results = run_regression_models()
    for model, scores in results.items():
        print(f"{model}: MSE={scores['MSE']:.2f}, R2={scores['R2']:.2f}")
