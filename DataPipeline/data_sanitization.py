"""Limpeza das fontes e geração do arquivo oficial ``Dados/clean_data.csv``."""
from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from DataPipeline.config import *
from MLOps import storage


def add_proxy(df: pd.DataFrame) -> pd.DataFrame:
    org=df.get("ORGANIZATION_TYPE", pd.Series(index=df.index,dtype="object"))
    income=df.get("NAME_INCOME_TYPE", pd.Series(index=df.index,dtype="object"))
    df[SMALL_BUSINESS_COL]=(org.isin(SMALL_BUSINESS_ORGANIZATION_TYPES)|income.isin(SMALL_BUSINESS_INCOME_TYPES)).astype("int8")
    return df


def clean_application() -> pd.DataFrame:
    df=storage.read_csv(RAW_DATA_PATH, low_memory=False)
    df.columns=[str(c).strip() for c in df.columns]
    if "DAYS_EMPLOYED" in df: df["DAYS_EMPLOYED"]=df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY,np.nan)
    df=df.drop_duplicates(subset=[ID_COL],keep="first")
    df=add_proxy(df)
    storage.write_csv(df,CLEAN_DATA_PATH)
    return df


def clean_bureau() -> pd.DataFrame:
    df=storage.read_csv(RAW_BUREAU_PATH,low_memory=False).drop_duplicates()
    if "SK_ID_BUREAU" in df: df=df.drop_duplicates("SK_ID_BUREAU")
    if "AMT_CREDIT_SUM_DEBT" in df: df["AMT_CREDIT_SUM_DEBT"]=df["AMT_CREDIT_SUM_DEBT"].clip(lower=0)
    storage.write_csv(df,CLEAN_BUREAU_PATH)
    return df


def clean_previous() -> pd.DataFrame:
    df=storage.read_csv(RAW_PREVIOUS_PATH,low_memory=False).drop_duplicates()
    if "SK_ID_PREV" in df: df=df.drop_duplicates("SK_ID_PREV")
    for col in ["DAYS_FIRST_DRAWING","DAYS_FIRST_DUE","DAYS_LAST_DUE_1ST_VERSION","DAYS_LAST_DUE","DAYS_TERMINATION"]:
        if col in df: df[col]=df[col].replace(DAYS_EMPLOYED_ANOMALY,np.nan)
    storage.write_csv(df,CLEAN_PREVIOUS_PATH)
    return df


def run() -> dict:
    app=clean_application(); bureau=clean_bureau(); prev=clean_previous()
    report={
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "clean_data":{"rows":len(app),"columns":app.shape[1],"default_rate":float(app[TARGET_COL].mean()),"small_business_rows":int(app[SMALL_BUSINESS_COL].sum())},
      "clean_bureau":{"rows":len(bureau),"columns":bureau.shape[1]},
      "clean_previous_application":{"rows":len(prev),"columns":prev.shape[1]},
    }
    storage.write_json(report,"reports/data_sanitization_report.json")
    print(json.dumps(report,ensure_ascii=False,indent=2)); return report

if __name__=="__main__": run()
