from sklearn.datasets import load_digits
import joblib

def run_inference():
    model = joblib.load("model_train.pkl")
    digits = load_digits()
    X, y = digits.data, digits.target
    preds = model.predict(X)
    print("Predictions:", preds[:10])

if __name__ == "__main__":
    run_inference()

