# Arquitetura da solução

## Visão geral

```text
Landing/
  └── CSVs do Kaggle
        ↓
Airflow
  ├── 00_check_inputs
  ├── 01_ingest_raw_data
  ├── 02_clean_data
  ├── 03_feature_aggregation
  ├── 04_build_abt
  ├── 05_train_model
  └── 06_score_sample
        ↓
MinIO / storage lógico
  ├── Dados/raw_data.csv
  ├── Dados/clean_data.csv
  ├── Dados/abt.csv
  ├── Model/model.pkl
  ├── Model/metrics.json
  └── reports/*
        ↓
Streamlit: scoring, analytics, matriz de confusão e explicabilidade
```

## Objetivo

A arquitetura existe para transformar as bases brutas do Kaggle em uma ABT modelável, treinar o modelo de risco e disponibilizar a inferência em uma interface visual simples.
