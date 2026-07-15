from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from MLOps import pipeline_orchestration as p

with DAG(
    "home_credit_empresarios_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["home-credit", "individual", "minio"],
) as dag:
    t0 = PythonOperator(task_id="00_check_inputs", python_callable=p.check_inputs)
    t1 = PythonOperator(task_id="01_ingest_raw_data", python_callable=p.ingest)
    t2 = PythonOperator(task_id="02_clean_data", python_callable=p.clean)
    t3 = PythonOperator(task_id="03_feature_aggregation", python_callable=p.aggregate)
    t4 = PythonOperator(task_id="04_build_abt", python_callable=p.build_abt)
    t5 = PythonOperator(task_id="05_train_model", python_callable=p.train)
    t6 = PythonOperator(task_id="06_score_sample", python_callable=p.score)

    t0 >> t1 >> t2 >> t3 >> t4 >> t5 >> t6
