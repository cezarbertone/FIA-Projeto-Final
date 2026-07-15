"""Serviço de predição do score de risco de crédito.

Uso em Python:
    from Model.predict import predict
    resultado = predict({"AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 300000, ...})

A função aceita dict, lista de dicts ou DataFrame, recria as features derivadas,
alinha as colunas ao modelo treinado e retorna probabilidade + decisão.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from MLOps import storage

_cfg_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("model_config", _cfg_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DECISION_THRESHOLD = config.RISK_THRESHOLD


def load_bundle(model_path: str | None = None) -> dict:
    """Carrega o bundle serializado em ``Model/model.pkl``."""
    return storage.read_pickle(model_path or config.MODEL_PATH)


def _as_dataframe(records: Union[dict, list, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(records, dict):
        return pd.DataFrame([records])
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, pd.DataFrame):
        return records.copy()
    raise TypeError("records deve ser dict, lista de dicts ou pandas.DataFrame")


def prepare_features(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Replica as features derivadas da ABT e garante o schema esperado pelo modelo."""
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(x.columns):
        x["CREDIT_INCOME_RATIO"] = x["AMT_CREDIT"] / x["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(x.columns):
        x["ANNUITY_INCOME_RATIO"] = x["AMT_ANNUITY"] / x["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(x.columns):
        x["ANNUITY_CREDIT_RATIO"] = x["AMT_ANNUITY"] / x["AMT_CREDIT"].replace(0, np.nan)
    if "DAYS_BIRTH" in x.columns:
        x["AGE_YEARS"] = -pd.to_numeric(x["DAYS_BIRTH"], errors="coerce") / 365.25
    if "DAYS_EMPLOYED" in x.columns:
        x["EMPLOYED_YEARS"] = -pd.to_numeric(x["DAYS_EMPLOYED"], errors="coerce") / 365.25

    # Nova solicitação sem histórico: agregações entram como zero. Demais campos ausentes
    # entram como NaN e são imputados pelo Pipeline salvo no model.pkl.
    for col in bundle["feature_columns"]:
        if col not in x.columns:
            x[col] = 0 if col.startswith(("BUREAU_", "PREV_", "INSTAL_")) else np.nan
    return x[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan)


def predict(
    records: Union[dict, list, pd.DataFrame],
    threshold: float | None = None,
    model_path: str | None = None,
) -> pd.DataFrame:
    """Gera probabilidade de inadimplência e decisão para uma ou mais solicitações."""
    df_in = _as_dataframe(records)
    bundle = load_bundle(model_path)
    X = prepare_features(df_in, bundle)
    probability = bundle["model"].predict_proba(X)[:, 1]
    decision_threshold = float(bundle.get("threshold", DECISION_THRESHOLD) if threshold is None else threshold)

    result = pd.DataFrame(index=df_in.index)
    if config.ID_COL in df_in.columns:
        result[config.ID_COL] = df_in[config.ID_COL].values
    result["PD_DEFAULT"] = probability
    result["THRESHOLD"] = decision_threshold
    result["CREDIT_DECISION"] = np.where(
        probability < decision_threshold,
        "CONCEDER",
        "REVISAR / NÃO CONCEDER",
    )
    result["ACTION_SUGGESTION"] = np.select(
        [
            probability < decision_threshold * 0.60,
            probability < decision_threshold,
            probability < min(1.0, decision_threshold + 0.15),
        ],
        [
            "Aprovação automática dentro da política.",
            "Aprovar com validação documental e limite conservador.",
            "Revisão manual: risco próximo ou acima do limite.",
        ],
        default="Não conceder automaticamente; encaminhar para análise de risco.",
    )
    return result.reset_index(drop=True)


def score_dataframe(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Compatibilidade com Streamlit: devolve entrada + colunas de scoring."""
    scored = predict(df, threshold=threshold)
    return pd.concat([df.reset_index(drop=True), scored], axis=1)


def score_sample(n: int = 1000, threshold: float | None = None) -> pd.DataFrame:
    """Escora amostra da ABT e persiste ``reports/scored_abt_sample.csv``."""
    df = storage.read_csv(config.ABT_PATH, low_memory=False)
    sample = df.sample(min(n, len(df)), random_state=config.RANDOM_STATE)
    scored = score_dataframe(sample, threshold=threshold)
    storage.write_csv(scored, "reports/scored_abt_sample.csv")
    return scored


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Score de risco Home Credit Empresários")
    parser.add_argument("--input", help="CSV de solicitações. Omitir para usar a ABT.")
    parser.add_argument("--n", type=int, default=5, help="Quantidade de linhas")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    if args.input:
        source = pd.read_csv(args.input, nrows=args.n, low_memory=False)
        output = predict(source, threshold=args.threshold)
    else:
        output = score_sample(args.n, threshold=args.threshold)
    print(output.to_string(index=False))


if __name__ == "__main__":
    _cli()
