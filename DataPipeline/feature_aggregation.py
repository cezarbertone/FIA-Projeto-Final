"""Agregação das fontes auxiliares no nível ``SK_ID_CURR``.

A ABT possui uma linha por solicitação. Bureau, pedidos anteriores e parcelas têm
múltiplas linhas por cliente e precisam ser resumidos antes do join.

Responsabilidade deste módulo: somente agregação e criação de features. A limpeza
de bureau/previous_application é feita em ``data_sanitization.py``.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from DataPipeline import config
from MLOps import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def aggregate_bureau(path: str = config.CLEAN_BUREAU_PATH) -> pd.DataFrame:
    """Histórico externo: volume, situação, dívida, atraso e recência."""
    columns = [
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG",
    ]
    bureau = storage.read_csv(path, low_memory=False)
    available = [c for c in columns if c in bureau.columns]
    bureau = bureau[available].copy()
    LOGGER.info("clean_bureau: %s linhas | %s clientes", f"{len(bureau):,}", f"{bureau[config.ID_COL].nunique():,}")

    bureau["_active"] = (bureau.get("CREDIT_ACTIVE") == "Active").astype("int8")
    bureau["_closed"] = (bureau.get("CREDIT_ACTIVE") == "Closed").astype("int8")

    agg_spec = {
        "BUREAU_CREDIT_COUNT": ("SK_ID_BUREAU", "count"),
        "BUREAU_ACTIVE_COUNT": ("_active", "sum"),
        "BUREAU_CLOSED_COUNT": ("_closed", "sum"),
    }
    optional = {
        "BUREAU_AMT_CREDIT_SUM_TOTAL": ("AMT_CREDIT_SUM", "sum"),
        "BUREAU_AMT_CREDIT_SUM_MEAN": ("AMT_CREDIT_SUM", "mean"),
        "BUREAU_AMT_DEBT_TOTAL": ("AMT_CREDIT_SUM_DEBT", "sum"),
        "BUREAU_DAY_OVERDUE_MAX": ("CREDIT_DAY_OVERDUE", "max"),
        "BUREAU_DAY_OVERDUE_MEAN": ("CREDIT_DAY_OVERDUE", "mean"),
        "BUREAU_AMT_OVERDUE_TOTAL": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
        "BUREAU_DAYS_CREDIT_MIN": ("DAYS_CREDIT", "min"),
        "BUREAU_DAYS_CREDIT_MAX": ("DAYS_CREDIT", "max"),
        "BUREAU_DAYS_CREDIT_MEAN": ("DAYS_CREDIT", "mean"),
        "BUREAU_CNT_PROLONG_TOTAL": ("CNT_CREDIT_PROLONG", "sum"),
    }
    agg_spec.update({name: spec for name, spec in optional.items() if spec[0] in bureau.columns})
    result = bureau.groupby(config.ID_COL).agg(**agg_spec)
    result["BUREAU_ACTIVE_RATIO"] = result["BUREAU_ACTIVE_COUNT"] / result["BUREAU_CREDIT_COUNT"].replace(0, np.nan)
    if {"BUREAU_AMT_DEBT_TOTAL", "BUREAU_AMT_CREDIT_SUM_TOTAL"}.issubset(result.columns):
        result["BUREAU_DEBT_CREDIT_RATIO"] = result["BUREAU_AMT_DEBT_TOTAL"] / result["BUREAU_AMT_CREDIT_SUM_TOTAL"].replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan).reset_index()
    storage.write_csv(result, config.FEATURES_BUREAU_PATH)
    LOGGER.info("bureau agregado: %s features | %s clientes", result.shape[1] - 1, f"{len(result):,}")
    return result


def aggregate_previous_application(path: str = config.CLEAN_PREVIOUS_PATH) -> pd.DataFrame:
    """Histórico interno: volume, aprovação/recusa, valores, prazo e recência."""
    columns = [
        "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_CREDIT",
        "AMT_APPLICATION", "AMT_DOWN_PAYMENT", "DAYS_DECISION", "CNT_PAYMENT",
    ]
    previous = storage.read_csv(path, low_memory=False)
    previous = previous[[c for c in columns if c in previous.columns]].copy()
    LOGGER.info("clean_previous_application: %s linhas | %s clientes", f"{len(previous):,}", f"{previous[config.ID_COL].nunique():,}")

    previous["_approved"] = (previous.get("NAME_CONTRACT_STATUS") == "Approved").astype("int8")
    previous["_refused"] = (previous.get("NAME_CONTRACT_STATUS") == "Refused").astype("int8")
    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(previous.columns):
        previous["_credit_app_ratio"] = previous["AMT_CREDIT"] / previous["AMT_APPLICATION"].replace(0, np.nan)

    agg_spec = {
        "PREV_APP_COUNT": ("SK_ID_PREV", "count"),
        "PREV_APPROVED_COUNT": ("_approved", "sum"),
        "PREV_REFUSED_COUNT": ("_refused", "sum"),
    }
    optional = {
        "PREV_AMT_CREDIT_MEAN": ("AMT_CREDIT", "mean"),
        "PREV_AMT_CREDIT_TOTAL": ("AMT_CREDIT", "sum"),
        "PREV_AMT_APPLICATION_MEAN": ("AMT_APPLICATION", "mean"),
        "PREV_CREDIT_APP_RATIO_MEAN": ("_credit_app_ratio", "mean"),
        "PREV_AMT_DOWN_PAYMENT_MEAN": ("AMT_DOWN_PAYMENT", "mean"),
        "PREV_DAYS_DECISION_MAX": ("DAYS_DECISION", "max"),
        "PREV_DAYS_DECISION_MIN": ("DAYS_DECISION", "min"),
        "PREV_CNT_PAYMENT_MEAN": ("CNT_PAYMENT", "mean"),
    }
    agg_spec.update({name: spec for name, spec in optional.items() if spec[0] in previous.columns})
    result = previous.groupby(config.ID_COL).agg(**agg_spec)
    result["PREV_APPROVAL_RATE"] = result["PREV_APPROVED_COUNT"] / result["PREV_APP_COUNT"].replace(0, np.nan)
    result["PREV_REFUSED_RATE"] = result["PREV_REFUSED_COUNT"] / result["PREV_APP_COUNT"].replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan).reset_index()
    storage.write_csv(result, config.FEATURES_PREVIOUS_PATH)
    LOGGER.info("previous agregado: %s features | %s clientes", result.shape[1] - 1, f"{len(result):,}")
    return result


def _aggregate_installment_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    required = [config.ID_COL, "SK_ID_PREV", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    missing = [c for c in required if c not in chunk.columns]
    if missing:
        raise ValueError(f"installments_payments sem colunas obrigatórias: {missing}")

    data = chunk[required].copy()
    data["_delay"] = data["DAYS_ENTRY_PAYMENT"] - data["DAYS_INSTALMENT"]
    data["_dpd"] = data["_delay"].clip(lower=0)
    data["_dbd"] = (-data["_delay"]).clip(lower=0)
    data["_payment_diff"] = data["AMT_PAYMENT"] - data["AMT_INSTALMENT"]
    data["_payment_ratio"] = data["AMT_PAYMENT"] / data["AMT_INSTALMENT"].replace(0, np.nan)
    data["_late"] = (data["_delay"] > 0).astype("int8")
    data["_late_30d"] = (data["_delay"] > 30).astype("int8")
    data["_early"] = (data["_delay"] < 0).astype("int8")
    data["_partial"] = (data["AMT_PAYMENT"] < data["AMT_INSTALMENT"]).astype("int8")

    grouped = data.groupby(config.ID_COL)
    return grouped.agg(
        INSTAL_COUNT=("SK_ID_PREV", "size"),
        INSTAL_PREV_NUNIQUE=("SK_ID_PREV", "nunique"),
        INSTAL_DELAY_SUM=("_delay", "sum"),
        INSTAL_DELAY_COUNT=("_delay", "count"),
        INSTAL_DELAY_MAX=("_delay", "max"),
        INSTAL_DELAY_MIN=("_delay", "min"),
        INSTAL_DPD_MAX=("_dpd", "max"),
        INSTAL_DBD_SUM=("_dbd", "sum"),
        INSTAL_LATE_COUNT=("_late", "sum"),
        INSTAL_LATE_30D_COUNT=("_late_30d", "sum"),
        INSTAL_EARLY_COUNT=("_early", "sum"),
        INSTAL_PARTIAL_COUNT=("_partial", "sum"),
        INSTAL_PAYMENT_RATIO_SUM=("_payment_ratio", "sum"),
        INSTAL_PAYMENT_RATIO_COUNT=("_payment_ratio", "count"),
        INSTAL_PAYMENT_RATIO_MIN=("_payment_ratio", "min"),
        INSTAL_PAYMENT_DIFF_SUM=("_payment_diff", "sum"),
        INSTAL_PAYMENT_DIFF_COUNT=("_payment_diff", "count"),
        INSTAL_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        INSTAL_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
    ).reset_index()


def aggregate_installments(path: str = config.RAW_INSTALLMENTS_PATH) -> pd.DataFrame:
    """Comportamento de parcelas processado em chunks para controlar uso de memória."""
    local_file = storage.materialize(path)
    usecols = [config.ID_COL, "SK_ID_PREV", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    read_args = {"usecols": usecols, "chunksize": config.INSTALLMENTS_CHUNKSIZE}
    if config.MAX_ROWS_INSTALLMENTS:
        read_args["nrows"] = config.MAX_ROWS_INSTALLMENTS

    partials = []
    for number, chunk in enumerate(pd.read_csv(local_file, **read_args), start=1):
        partials.append(_aggregate_installment_chunk(chunk))
        LOGGER.info("installments chunk %s: %s linhas", number, f"{len(chunk):,}")

    if not partials:
        result = pd.DataFrame(columns=[config.ID_COL])
    else:
        combined = pd.concat(partials, ignore_index=True)
        sum_columns = [c for c in combined.columns if c not in {config.ID_COL, "INSTAL_DELAY_MAX", "INSTAL_DELAY_MIN", "INSTAL_DPD_MAX", "INSTAL_PAYMENT_RATIO_MIN"}]
        aggregations = {c: "sum" for c in sum_columns}
        aggregations.update({
            "INSTAL_DELAY_MAX": "max",
            "INSTAL_DELAY_MIN": "min",
            "INSTAL_DPD_MAX": "max",
            "INSTAL_PAYMENT_RATIO_MIN": "min",
        })
        result = combined.groupby(config.ID_COL).agg(aggregations).reset_index()
        delay_den = result["INSTAL_DELAY_COUNT"].replace(0, np.nan)
        result["INSTAL_DELAY_MEAN"] = result["INSTAL_DELAY_SUM"] / delay_den
        result["INSTAL_DBD_MEAN"] = result["INSTAL_DBD_SUM"] / delay_den
        result["INSTAL_LATE_PAYMENT_RATIO"] = result["INSTAL_LATE_COUNT"] / delay_den
        result["INSTAL_LATE_30D_RATIO"] = result["INSTAL_LATE_30D_COUNT"] / delay_den
        result["INSTAL_EARLY_PAYMENT_RATIO"] = result["INSTAL_EARLY_COUNT"] / delay_den
        result["INSTAL_PARTIAL_PAYMENT_RATIO"] = result["INSTAL_PARTIAL_COUNT"] / result["INSTAL_COUNT"].replace(0, np.nan)
        result["INSTAL_PAYMENT_RATIO_MEAN"] = result["INSTAL_PAYMENT_RATIO_SUM"] / result["INSTAL_PAYMENT_RATIO_COUNT"].replace(0, np.nan)
        result["INSTAL_PAYMENT_DIFF_MEAN"] = result["INSTAL_PAYMENT_DIFF_SUM"] / result["INSTAL_PAYMENT_DIFF_COUNT"].replace(0, np.nan)

    result = result.replace([np.inf, -np.inf], np.nan)
    storage.write_csv(result, config.FEATURES_INSTALLMENTS_PATH)
    LOGGER.info("installments agregado: %s features | %s clientes", max(0, result.shape[1] - 1), f"{len(result):,}")
    return result


def run() -> dict:
    bureau = aggregate_bureau()
    previous = aggregate_previous_application()
    installments = aggregate_installments()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bureau_shape": list(bureau.shape),
        "previous_shape": list(previous.shape),
        "installments_shape": list(installments.shape),
    }
    storage.write_json(report, "reports/feature_aggregation_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
