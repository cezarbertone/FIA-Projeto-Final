# Dados oficiais gerados pelo pipeline

Após a execução da DAG, esta pasta contém também uma réplica local dos objetos oficiais mantidos no MinIO:

- `raw_data.csv`
- `clean_data.csv`
- `abt.csv`

Os arquivos não acompanham o ZIP porque são grandes e são reconstruídos a partir de `Landing/`.
