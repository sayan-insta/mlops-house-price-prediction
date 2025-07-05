#  MLOps House Price Prediction

This project implements a **House Price Prediction** system using the **California Housing dataset** and demonstrates a simple **MLOps pipeline using GitHub Actions**.

The repository includes:
- Data loading
- Preprocessing
- Model training (Linear Regression)
- Model evaluation
- Automated CI/CD workflow

---

## Project Structure

```text
.
├── data/                  # Data folder (no CSV needed, using sklearn dataset)
├── regression.py          # Main regression training and evaluation script
├── utils.py               # Utility functions for data loading and splitting
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .github/
    └── workflows/
        └── ci.yml         # GitHub Actions workflow file

