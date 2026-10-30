# Bike Demand Prediction — ML Mid-Project

A machine-learning project that predicts the number of rented bikes (regression) and classifies expected demand as **Low, Medium, or High** (classification). The project includes data preparation, model training/evaluation, saved models, a notebook, and a dark-purple FastAPI website.

## 1. Project structure and order

```text
ml-project/
├── .github/
│   └── workflows/ci.yml           # Automated checks (GitHub Actions)
├── config/
│   ├── config.yaml                # Project settings and paths
│   └── .env.example               # Safe template; no real secrets
├── data/
│   ├── raw/train.csv              # Original dataset
│   └── processed/                 # Prepared train/test data
├── models/                        # Saved estimators, metrics, and plots
├── notebooks/
│   ├── scratchpad/                # Temporary exploratory notebooks
│   ├── 01_eda.ipynb               # Explore and understand the data
│   └── 02_modeling.ipynb          # Compare models and review results
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── data_ingestion.py      # Load and split data
│   │   ├── data_transform.py      # Prepare input features
│   │   └── model_trainer.py       # Train and evaluate models
│   ├── models/
│   │   ├── regression/            # Regression model code
│   │   └── classification/        # Classification model code
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Full training workflow
│   │   └── predict_pipeline.py    # Prediction workflow
│   ├── utils/
│   │   └── helpers.py             # Reusable helper functions
│   ├── logger.py                 # Logging setup
│   └── exception.py              # Custom exception handling
├── tests/
│   ├── test_data.py              # Data checks
│   └── test_model.py             # Model checks
├── static/index.html              # Dark-purple website frontend
├── app.py                         # Website backend and prediction APIs
├── Dockerfile                     # Container setup
├── Makefile                       # Shortcut commands
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignore local data, models, caches, secrets
└── README.md                      # Project guide
```

The ANN (Artificial Neural Network) and KNN (K-Nearest Neighbours) models have been removed from the training workflow, saved model files, and website model selectors. The remaining project features and website design are retained.

## 2. Regression models

Regression predicts a **numeric bike rental count**. The code for each model is kept in `src/models/regression/`.

1. **Decision Tree Regressor** — learns decision rules from the input features.
2. **Support Vector Regression (SVR)** — uses a support-vector method to predict a numeric count.
3. **Random Forest Regressor** — combines multiple decision trees to make a more stable prediction.

Regression is evaluated using **R²** (higher is better), **MAE** (lower is better), and **RMSE** (lower is better).

## 3. Classification models

Classification predicts a demand category: **Low, Medium, or High**. The code for each model is kept in `src/models/classification/`.

1. **Decision Tree Classifier** — learns rules to choose a demand category.
2. **Gaussian Naive Bayes** — estimates the probability of each demand category.
3. **Support Vector Classifier (SVC)** — separates the demand categories using a support-vector method.
4. **Random Forest Classifier** — combines multiple decision trees to choose a category.

Naive Bayes is classification-only. Classification is evaluated using **accuracy**, **macro-F1**, and a confusion matrix.

## 4. Data and evaluation

- Dataset: Bike Sharing Demand hourly rental data in `data/raw/train.csv`.
- Regression target: `count`, the total number of rentals.
- Classification target: `count` converted to Low, Medium, and High using cutoffs calculated from training data only.
- Data split: earlier records are used for training and the latest records for testing, based on datetime.
- `casual` and `registered` are not used as input features because they directly contribute to the total `count` and would cause target leakage.
- Feature preparation is shared by training and the website so inputs are processed consistently.
- Results and model comparisons are saved in `models/metrics.json` and `models/model_comparison.json`. These files are refreshed when training runs; use the newly generated results for your report.

## 5. Run the project (Windows PowerShell)

Run these commands from the `ml-project` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.pipeline.train_pipeline
python -m pytest -q
python -m uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000` in your browser. If port 8000 is already in use or blocked, stop the process using it or choose another port, for example `--port 8001`.

## 6. Important files

- `src/components/model_trainer.py`: central training, tuning, evaluation, and model saving.
- `src/models/regression/`: separate regression estimator code.
- `src/models/classification/`: separate classification estimator code.
- `notebooks/02_modeling.ipynb`: model comparison and evaluation notebook.
- `app.py` and `static/index.html`: prediction backend and website.
- `models/metrics.json`: latest model scores after training.
