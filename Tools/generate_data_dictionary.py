from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
from DataPipeline.config import RAW_DATA_PATH,CLEAN_DATA_PATH,ABT_PATH
from MLOps import storage

DESCRIPTIONS={
"SMALL_BUSINESS_PROXY":"Proxy de pequeno empresário/autônomo.",
"CREDIT_INCOME_RATIO":"Crédito dividido pela renda.",
"ANNUITY_INCOME_RATIO":"Anuidade/parcela dividida pela renda.",
"ANNUITY_CREDIT_RATIO":"Anuidade dividida pelo crédito.",
"INSTAL_DELAY_MEAN":"Atraso médio histórico de pagamento, em dias.",
"INSTAL_DPD_MAX":"Maior atraso histórico em dias.",
"INSTAL_LATE_PAYMENT_RATIO":"Proporção de parcelas pagas em atraso.",
"INSTAL_LATE_30D_RATIO":"Proporção de parcelas com atraso superior a 30 dias.",
"INSTAL_PARTIAL_PAYMENT_RATIO":"Proporção de pagamentos parciais.",
}

def table(df):
    lines=["| Coluna | Tipo | % Nulos | Únicos | Descrição |","|---|---:|---:|---:|---|"]
    for c in df.columns:
        desc=DESCRIPTIONS.get(c,"Feature original ou agregada do conjunto Home Credit.")
        lines.append(f"| `{c}` | {df[c].dtype} | {df[c].isna().mean():.1%} | {df[c].nunique():,} | {desc} |")
    return "\n".join(lines)

def run():
    for key,name,title in [(RAW_DATA_PATH,"dicionario_raw_data.md","raw_data.csv"),(CLEAN_DATA_PATH,"dicionario_clean_data.md","clean_data.csv"),(ABT_PATH,"dicionario_abt.md","abt.csv")]:
        if not storage.exists(key):
            print(f"Ignorado: {key} ainda não existe"); continue
        df=storage.read_csv(key,low_memory=False)
        text=f"# Dicionário — {title}\n\nDimensão: {len(df):,} linhas × {df.shape[1]} colunas.\n\n"+table(df)+"\n"
        (ROOT/"DataPipeline"/name).write_text(text,encoding="utf-8")
        print(f"Gerado: DataPipeline/{name}")
if __name__=="__main__":run()
