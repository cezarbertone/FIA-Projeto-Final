"""Configuração central da modelagem do score de risco de crédito."""
from __future__ import annotations
import os

ABT_PATH = "Dados/abt.csv"
MODEL_PATH = "Model/model.pkl"
METRICS_PATH = "Model/metrics.json"
REFERENCE_PROFILE_PATH = "Model/reference_profile.json"
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

TEST_SIZE = float(os.getenv("MODEL_TEST_SIZE", "0.20"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.50"))
CV_FOLDS = int(os.getenv("MODEL_CV_FOLDS", "5"))
SEARCH_SAMPLE_FRAC = float(os.getenv("MODEL_SEARCH_SAMPLE_FRAC", "0.30"))
ANALYSIS_THRESHOLDS = [float(x.strip()) for x in os.getenv("MODEL_ANALYSIS_THRESHOLDS", "0.30,0.50,0.70").split(",")]

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
LOGISTIC_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}
SEARCH_GRIDS = {
    "gradient_boosting": {
        "model__n_estimators": [150, 200],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
    },
    "random_forest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [8, 12],
        "model__min_samples_leaf": [20, 50],
    },
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
}
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "true").lower() == "true"
SHAP_SAMPLE_ROWS = int(os.getenv("SHAP_SAMPLE_ROWS", "500"))
PERMUTATION_SAMPLE_ROWS = int(os.getenv("PERMUTATION_SAMPLE_ROWS", "5000"))
PERMUTATION_REPEATS = int(os.getenv("PERMUTATION_REPEATS", "3"))
