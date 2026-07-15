"""Treinamento do score de risco para pequenos empresários/autônomos.

Fluxo metodológico:
1. Holdout estratificado de 20%, mantido intocado.
2. Comparação de Logistic Regression, Random Forest e Gradient Boosting por CV-AUC.
3. GridSearchCV somente no algoritmo vencedor.
4. Fit final do vencedor ajustado no treino completo.
5. Avaliação única no holdout e persistência de modelo, métricas e relatórios.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from MLOps import storage

_cfg_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("model_config", _cfg_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)
NEEDS_SAMPLE_WEIGHT = {"gradient_boosting"}


def build_preprocessors(X: pd.DataFrame):
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    tree_numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    linear = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
    tree = ColumnTransformer([
        ("numeric", tree_numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
    return linear, tree, numeric, categorical


def build_candidates(linear_preprocessor, tree_preprocessor):
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", linear_preprocessor),
            ("model", LogisticRegression(**config.LOGISTIC_PARAMS)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", GradientBoostingClassifier(**config.GRADIENT_BOOSTING_PARAMS)),
        ]),
    }


def fit_model(model, name: str, X: pd.DataFrame, y: pd.Series):
    """Fit centralizado; Gradient Boosting recebe pesos por não ter class_weight."""
    if name in NEEDS_SAMPLE_WEIGHT:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def fit_params(name: str, y: pd.Series) -> dict:
    if name in NEEDS_SAMPLE_WEIGHT:
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def confusion_metrics(y_true, y_score, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "approval_rate": float((pred == 0).mean()),
        "review_or_deny_rate": float((pred == 1).mean()),
        "recall_default": float(recall_score(y_true, pred, zero_division=0)),
        "precision_default": float(precision_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def subsample(X, y, frac):
    if not frac or frac >= 1:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=config.RANDOM_STATE
    )
    return Xs, ys


def cross_validate_candidate(name, model, X, y, cv):
    scores = []
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        fitted = fit_model(clone(model), name, X.iloc[train_idx], y.iloc[train_idx])
        proba = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        auc = float(roc_auc_score(y.iloc[valid_idx], proba))
        scores.append(auc)
        LOGGER.info("%s fold %s/%s: AUC=%.4f", name, fold, config.CV_FOLDS, auc)
    return scores


def evaluate(model, X_hold, y_hold, threshold):
    proba = model.predict_proba(X_hold)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, {
        "rows": int(len(X_hold)),
        "default_rate": float(y_hold.mean()),
        "auc_roc": float(roc_auc_score(y_hold, proba)),
        "ks": ks_statistic(y_hold, proba),
        "average_precision": float(average_precision_score(y_hold, proba)),
        "recall_default": float(recall_score(y_hold, pred, zero_division=0)),
        "precision_default": float(precision_score(y_hold, pred, zero_division=0)),
        "f1_default": float(f1_score(y_hold, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_hold, pred)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_metrics(y_hold, proba, threshold),
    }


def feature_names(pipe):
    try:
        return pipe.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        return []


def save_explainability(pipe, X_hold, y_hold):
    names = feature_names(pipe)
    estimator = pipe.named_steps["model"]
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    if values is not None:
        n = min(len(names), len(values))
        native = pd.DataFrame({"feature": names[:n], "importance": np.asarray(values)[:n]})
        native["abs_value"] = native["importance"].abs()
        storage.write_csv(native.sort_values("abs_value", ascending=False), "reports/feature_importance.csv")

    X_perm, y_perm = X_hold, y_hold
    if config.PERMUTATION_SAMPLE_ROWS and len(X_hold) > config.PERMUTATION_SAMPLE_ROWS:
        X_perm, _, y_perm, _ = train_test_split(
            X_hold, y_hold, train_size=config.PERMUTATION_SAMPLE_ROWS,
            stratify=y_hold, random_state=config.RANDOM_STATE
        )
    result = permutation_importance(
        pipe, X_perm, y_perm, n_repeats=config.PERMUTATION_REPEATS,
        scoring="roc_auc", random_state=config.RANDOM_STATE, n_jobs=1
    )
    perm = pd.DataFrame({
        "feature": X_perm.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    storage.write_csv(perm, "reports/permutation_importance.csv")

    status = {"enabled": config.ENABLE_SHAP, "status": "not_run"}
    if config.ENABLE_SHAP:
        try:
            import shap
            sample = X_hold.sample(min(config.SHAP_SAMPLE_ROWS, len(X_hold)), random_state=config.RANDOM_STATE)
            transformed = pipe.named_steps["preprocessor"].transform(sample)
            explainer = shap.Explainer(estimator, transformed, feature_names=names)
            arr = np.asarray(explainer(transformed).values)
            if arr.ndim == 3:
                arr = arr[:, :, -1]
            shap_df = pd.DataFrame({
                "feature": names[:arr.shape[1]],
                "mean_abs_shap": np.abs(arr).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            storage.write_csv(shap_df, "reports/shap_importance.csv")
            status = {"enabled": True, "status": "ok", "sample_rows": int(len(sample))}
        except Exception as exc:
            status = {"enabled": True, "status": "failed", "error": str(exc)}
    storage.write_json(status, "reports/shap_status.json")


def save_reference_profile(X_train, y_train, p_train):
    edges = np.unique(np.quantile(p_train, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        edges = np.array([0.0, 0.5, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(p_train, bins=edges)[0].astype(float)
    selected = [c for c in [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "CREDIT_INCOME_RATIO",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "INSTAL_DELAY_MEAN", "INSTAL_LATE_PAYMENT_RATIO",
        "BUREAU_DEBT_CREDIT_RATIO",
    ] if c in X_train.columns]
    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(X_train)),
        "target_rate": float(y_train.mean()),
        "pd_histogram": {
            "edges": [None if np.isinf(v) else float(v) for v in edges],
            "proportions": (counts / counts.sum()).tolist(),
        },
        "features": {},
    }
    for col in selected:
        s = pd.to_numeric(X_train[col], errors="coerce")
        profile["features"][col] = {
            "missing_rate": float(s.isna().mean()),
            "mean": None if pd.isna(s.mean()) else float(s.mean()),
            "std": None if pd.isna(s.std()) else float(s.std()),
        }
    storage.write_json(profile, config.REFERENCE_PROFILE_PATH)


def train():
    LOGGER.info("Carregando ABT: %s", config.ABT_PATH)
    df = storage.read_csv(config.ABT_PATH, low_memory=False)
    X = df.drop(columns=[c for c in (config.ID_COL, config.TARGET_COL) if c in df.columns])
    y = df[config.TARGET_COL].astype(int)

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    X_select, y_select = subsample(X_train, y_train, config.SEARCH_SAMPLE_FRAC)
    LOGGER.info("Amostras=%s | Features=%s | Default=%.2f%%", len(df), X.shape[1], y.mean() * 100)
    LOGGER.info("Treino=%s | Holdout=%s | Seleção/Busca=%s", len(X_train), len(X_hold), len(X_select))

    linear_pre, tree_pre, numeric, categorical = build_preprocessors(X_train)
    candidates = build_candidates(linear_pre, tree_pre)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    comparison = []
    for name, model in candidates.items():
        fold_scores = cross_validate_candidate(name, model, X_select, y_select, cv)
        comparison.append({
            "model": name,
            "cv_auc_mean": float(np.mean(fold_scores)),
            "cv_auc_std": float(np.std(fold_scores, ddof=1)),
            "folds": config.CV_FOLDS,
            "fold_scores": ";".join(f"{v:.6f}" for v in fold_scores),
        })
    comparison_df = pd.DataFrame(comparison).sort_values("cv_auc_mean", ascending=False)
    storage.write_csv(comparison_df, "reports/model_comparison.csv")
    winner = str(comparison_df.iloc[0]["model"])
    LOGGER.info("Vencedor por CV-AUC: %s", winner)

    search = GridSearchCV(
        candidates[winner], config.SEARCH_GRIDS[winner], scoring="roc_auc",
        cv=cv, n_jobs=1, return_train_score=True
    )
    search.fit(X_select, y_select, **fit_params(winner, y_select))
    storage.write_csv(pd.DataFrame(search.cv_results_), "reports/grid_search_results.csv")
    LOGGER.info("GridSearch: best_score=%.4f | best_params=%s", search.best_score_, search.best_params_)

    final_model = fit_model(clone(search.best_estimator_), winner, X_train, y_train)
    proba, holdout = evaluate(final_model, X_hold, y_hold, config.RISK_THRESHOLD)

    metrics = {
        "best_model": winner,
        "selection": {
            "cv_folds": config.CV_FOLDS,
            "search_sample_frac": config.SEARCH_SAMPLE_FRAC,
            "candidate_cv": comparison,
            "best_params": search.best_params_,
            "best_grid_cv_auc": float(search.best_score_),
        },
        "holdout": holdout,
        "data": {
            "rows": int(len(df)), "features": int(X.shape[1]),
            "train_rows": int(len(X_train)), "holdout_rows": int(len(X_hold)),
            "target_rate": float(y.mean()),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    bundle = {
        "model": final_model,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "best_model": winner,
        "threshold": config.RISK_THRESHOLD,
    }
    storage.write_pickle(bundle, config.MODEL_PATH)
    storage.write_json(metrics, config.METRICS_PATH)

    ids = df.loc[X_hold.index, config.ID_COL].values if config.ID_COL in df.columns else X_hold.index
    storage.write_csv(pd.DataFrame({config.ID_COL: ids, "TARGET": y_hold.values, "PD_DEFAULT": proba}), "reports/holdout_predictions.csv")
    storage.write_csv(pd.DataFrame([confusion_metrics(y_hold, proba, t) for t in config.ANALYSIS_THRESHOLDS]), "reports/threshold_analysis.csv")
    fpr, tpr, thresholds = roc_curve(y_hold, proba)
    storage.write_csv(pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}), "reports/roc_curve_best_model.csv")
    save_explainability(final_model, X_hold, y_hold)
    save_reference_profile(X_train, y_train, final_model.predict_proba(X_train)[:, 1])

    LOGGER.info("Holdout: AUC=%.4f | KS=%.4f | Recall=%.4f", holdout["auc_roc"], holdout["ks"], holdout["recall_default"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    train()
