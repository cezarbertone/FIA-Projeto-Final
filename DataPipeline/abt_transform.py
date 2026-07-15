"""Geração de ``Dados/abt.csv``: uma linha por solicitação de pequeno empresário."""
# 1. lê Dados/clean_data.csv
# 2. filtra SMALL_BUSINESS_PROXY = 1
# 3. cria razões financeiras
# 4. junta features BUREAU_*
# 5. junta features PREV_*
# 6. junta features INSTAL_*
# 7. preenche histórico ausente com zero
# 8. remove colunas com muitos nulos
# 9. preserva variáveis importantes como EXT_SOURCE
# 10. grava Dados/abt.csv

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


def ratios(df):
    df=df.copy();
    if {"AMT_CREDIT","AMT_INCOME_TOTAL"}.issubset(df): df["CREDIT_INCOME_RATIO"]=df["AMT_CREDIT"]/df["AMT_INCOME_TOTAL"].replace(0,np.nan)
    if {"AMT_ANNUITY","AMT_INCOME_TOTAL"}.issubset(df): df["ANNUITY_INCOME_RATIO"]=df["AMT_ANNUITY"]/df["AMT_INCOME_TOTAL"].replace(0,np.nan)
    if {"AMT_ANNUITY","AMT_CREDIT"}.issubset(df): df["ANNUITY_CREDIT_RATIO"]=df["AMT_ANNUITY"]/df["AMT_CREDIT"].replace(0,np.nan)
    if "DAYS_BIRTH" in df: df["AGE_YEARS"]=(-df["DAYS_BIRTH"])/365.25
    if "DAYS_EMPLOYED" in df: df["EMPLOYED_YEARS"]=(-df["DAYS_EMPLOYED"])/365.25
    return df


def run():
    app=storage.read_csv(CLEAN_DATA_PATH,low_memory=False); source_rows=len(app)
    if FILTER_SMALL_BUSINESS: app=app[app[SMALL_BUSINESS_COL]==1].copy()
    abt=ratios(app)
    for path in (FEATURES_BUREAU_PATH,FEATURES_PREVIOUS_PATH,FEATURES_INSTALLMENTS_PATH):
        if storage.exists(path): abt=abt.merge(storage.read_csv(path),on=ID_COL,how="left")
    hist=[c for c in abt if c.startswith(("BUREAU_","PREV_","INSTAL_"))]
    if hist: abt[hist]=abt[hist].fillna(0)
    miss=abt.isna().mean(); protected={ID_COL,TARGET_COL,SMALL_BUSINESS_COL,*PROTECTED_NULL_COLUMNS}
    dropped=[c for c,r in miss.items() if r>NULL_DROP_THRESHOLD and c not in protected]
    abt=abt.drop(columns=dropped).replace([np.inf,-np.inf],np.nan)
    storage.write_csv(abt,ABT_PATH)
    report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"source_rows":source_rows,"small_business_rows":len(abt),"rows":len(abt),"columns":abt.shape[1],"features":abt.shape[1]-2,"default_rate":float(abt[TARGET_COL].mean()),"dropped_high_null_columns":dropped}
    storage.write_json(report,"reports/abt_report.json"); print(json.dumps(report,ensure_ascii=False,indent=2)); return report
if __name__=="__main__": run()
