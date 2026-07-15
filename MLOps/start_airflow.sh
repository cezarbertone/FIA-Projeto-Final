#!/usr/bin/env bash
set -euo pipefail

export AIRFLOW_HOME="${AIRFLOW_HOME:-/opt/airflow}"
mkdir -p "$AIRFLOW_HOME/logs" "$AIRFLOW_HOME/plugins"
rm -f "$AIRFLOW_HOME/airflow-webserver.pid"

echo "Inicializando metadata DB local do Airflow (SQLite)..."
airflow db migrate

echo "Criando usuário admin idempotente..."
airflow users create \
  --username "${AIRFLOW_ADMIN_USER:-airflow}" \
  --password "${AIRFLOW_ADMIN_PASSWORD:-airflow}" \
  --firstname Airflow \
  --lastname Admin \
  --role Admin \
  --email airflow@example.com || true

echo "Listando DAGs disponíveis..."
airflow dags list || true

echo "Iniciando scheduler em background..."
airflow scheduler &

echo "Iniciando webserver na porta 8080..."
exec airflow webserver --port 8080
