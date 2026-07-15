# MLOps — Arquitetura individual

## Componentes

| Componente | Papel |
|---|---|
| Airflow | orquestra as etapas em ordem e registra logs/status |
| MinIO | data lake e fonte de verdade de dados, modelos e relatórios |
| Streamlit | scoring, analytics, matriz de confusão e explicabilidade |
| Scikit-learn | transformação, seleção, tuning, treino e inferência |

A stack deliberadamente não usa Redis, Celery, Postgres ou FastAPI. O Airflow utiliza `SequentialExecutor + SQLite`, suficiente para uma demonstração local robusta e compatível com o entregável individual.

## Persistência direta no MinIO

Todos os scripts usam `MLOps/storage.py`. Com `STORAGE_BACKEND=minio`, chamadas como:

```python
storage.read_csv("Dados/abt.csv")
storage.write_pickle(modelo, "Model/model.pkl")
```

leem e gravam diretamente no bucket. Não há task `sync_minio`.

## Serviços

```text
minio       9000/9001
minio-init  cria o bucket
airflow     8080
streamlit   8501
```

## Fluxo orquestrado

```text
00_check_inputs
→ 01_ingest_raw_data
→ 02_clean_data
→ 03_feature_aggregation
→ 04_build_abt
→ 05_train_model
→ 06_score_sample
```

## Próximos passos de produção

- separar ambientes dev/homolog/prod;
- versionar objetos por data de referência;
- introduzir catálogo e lineage;
- registrar modelos em um model registry.
