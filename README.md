# Home Credit — Risco de Crédito para Pequenos Empresários

Projeto individual do MBA Pós Big Data e Analytics. A solução utiliza os dados do desafio **Home Credit Default Risk** para estimar a probabilidade de inadimplência de clientes com perfil próximo ao de pequenos empresários, autônomos e profissionais com renda comercial.

## 1. Problema de negócio

A instituição precisa decidir, no momento da solicitação, se concede crédito, encaminha o caso para revisão ou não concede automaticamente. O modelo entrega uma **probabilidade de default (PD)**; a política de crédito escolhe o threshold conforme seu apetite a risco.

- `TARGET = 0`: cliente sem dificuldade de pagamento.
- `TARGET = 1`: cliente com dificuldade de pagamento/default.
- Erro mais caro: **falso negativo**, isto é, aprovar um cliente que se tornará inadimplente.

### Recorte individual

A fonte é de pessoa física, portanto “pequeno empresário” é uma proxy analítica:

```text
SMALL_BUSINESS_PROXY =
    ORGANIZATION_TYPE == "Self-employed"
    OU NAME_INCOME_TYPE em ["Commercial associate", "Businessman"]
```

O filtro é aplicado na construção da ABT; o arquivo `clean_data.csv` preserva toda a base limpa e a flag da proxy para auditoria.

## 2. Fontes de dados

Coloque os quatro CSVs na pasta **`Landing/` da raiz**, fora da pasta `Dados`:

```text
Landing/
  application_train.csv
  bureau.csv
  previous_application.csv
  installments_payments.csv
```

Nomes com sufixo de download, como `application_train(1).csv`, também são reconhecidos. Arquivos `.rar` ou `.zip` precisam ser extraídos antes.

| Fonte | Papel |
|---|---|
| `application_train` | solicitação atual, cadastro, renda, crédito e `TARGET` |
| `bureau` | histórico externo em outras instituições |
| `previous_application` | pedidos anteriores na Home Credit |
| `installments_payments` | comportamento real de pagamento de parcelas |

## 3. Arquitetura de dados oficial

A estrutura pedida para a entrega é mantida em todos os backends:

```text
Landing/                 # arquivos recebidos; fora de Dados
Dados/raw_data.csv       # application_train promovido sem transformação
Dados/clean_data.csv     # application limpa + SMALL_BUSINESS_PROXY
Dados/abt.csv            # ABT filtrada e enriquecida, 1 linha por SK_ID_CURR
```

Arquivos auxiliares de processamento ficam em `Dados/_processing/`. Eles não são camadas oficiais; existem para limpar e agregar `bureau`, `previous_application` e `installments_payments` sem misturar seu nível de registro com os três entregáveis principais.

## 4. Persistência: local ou MinIO

A camada `MLOps/storage.py` mantém os mesmos caminhos lógicos e troca apenas o backend:

```text
STORAGE_BACKEND=local   → filesystem do projeto
STORAGE_BACKEND=minio   → object keys no bucket home-credit-empresarios
```

No Docker Compose, **MinIO é a fonte de verdade**: ingestão, limpeza, ABT, modelo, métricas e relatórios são lidos e gravados diretamente no bucket. Não existe uma etapa final de “sincronização”. Para facilitar a avaliação acadêmica, `MIRROR_LOCAL_OUTPUTS=true` replica também os três CSVs oficiais e os artefatos principais do modelo no filesystem local.

Principais objetos no MinIO e, para os itens oficiais, também no repositório local:

```text
Dados/raw_data.csv
Dados/clean_data.csv
Dados/abt.csv
Dados/_processing/...
Model/model.pkl
Model/metrics.json
Model/reference_profile.json
reports/...
```

## 5. Pipeline Airflow

A DAG `home_credit_empresarios_pipeline` usa `SequentialExecutor` e SQLite, mantendo o Airflow simples para execução local.

```text
00_check_inputs
→ 01_ingest_raw_data
→ 02_clean_data
→ 03_feature_aggregation
→ 04_build_abt
→ 05_train_model
→ 06_score_sample
```

### Responsabilidade de cada etapa

1. **Check inputs:** confirma os quatro CSVs na Landing.
2. **Ingest raw:** promove os arquivos para o MinIO; `application_train` vira `Dados/raw_data.csv`.
3. **Clean data:** gera `Dados/clean_data.csv` e auxiliares limpos.
4. **Feature aggregation:** cria `BUREAU_*`, `PREV_*` e `INSTAL_*` por `SK_ID_CURR`.
5. **Build ABT:** filtra a proxy de pequeno empresário e gera `Dados/abt.csv`.
6. **Train:** seleciona, ajusta e avalia o modelo.
7. **Score sample:** gera uma amostra de decisões para demonstração no Streamlit.

## 6. Treinamento do modelo

O desenho replica o fluxo metodológico do projeto em grupo:

1. `train_test_split` estratificado: **80% treino / 20% holdout**.
2. Comparação de **Logistic Regression**, **Random Forest** e **Gradient Boosting** por AUC-ROC média em **StratifiedKFold com 5 folds**, somente no treino.
3. `GridSearchCV` apenas no algoritmo vencedor.
4. Retreino do vencedor ajustado em todo o conjunto de treino.
5. Avaliação **uma única vez** no holdout intocado.

A seleção e o tuning usam 30% do treino para reduzir tempo; o fit final e o holdout usam todos os registros disponíveis.

### Desbalanceamento

- Logistic Regression e Random Forest: `class_weight="balanced"`.
- Gradient Boosting: `sample_weight` balanceado, pois o algoritmo não possui `class_weight`.

### Métricas

- **AUC-ROC:** capacidade de ordenar clientes do menor para o maior risco.
- **KS:** separação entre bons e maus pagadores.
- **Recall:** percentual de inadimplentes capturados.
- **Precision:** entre os clientes marcados como risco, quantos realmente inadimpliram.
- **Matriz de confusão:** VN, FP, FN e VP.
- **Threshold:** comparação automática em 0,30, 0,50 e 0,70 e simulação interativa no Streamlit.

### Artefatos gerados

```text
Model/model.pkl
Model/metrics.json
Model/reference_profile.json
reports/model_comparison.csv
reports/grid_search_results.csv
reports/holdout_predictions.csv
reports/threshold_analysis.csv
reports/roc_curve_best_model.csv
reports/feature_importance.csv
reports/permutation_importance.csv
reports/shap_importance.csv
```

## 7. Avaliação e explicabilidade

A avaliação usa três visões complementares:

1. **Importância nativa:** importância das árvores ou coeficientes do modelo linear.
2. **Permutation importance:** queda da AUC ao embaralhar cada variável original.
3. **SHAP:** contribuição média absoluta das features transformadas.

O notebook `Model/evaluation.ipynb` e o Streamlit apresentam resultados, curva ROC, matriz de confusão, thresholds e explicabilidade.

## 8. Serving em Streamlit

O projeto não usa FastAPI. O Streamlit oferece:

- layout visual baseado na V1.4.1, com contraste reforçado para melhor leitura;
- cliente histórico aleatório da ABT;
- formulário de nova solicitação simplificada;
- gauge da PD entre 0 e 1;
- threshold ajustável na barra lateral;
- decisão `CONCEDER` ou `REVISAR / NÃO CONCEDER`;
- resultados de treinamento;
- matriz de confusão interativa;
- importância nativa, permutation e SHAP;
- visualização dos objetos do MinIO.

Na entrada simplificada, campos não informados são imputados pelo pipeline. Agregações históricas ausentes são assumidas como zero, representando uma solicitação sem histórico disponível.

## 9. Como subir

Na raiz do projeto:

```powershell
docker compose -f .\MLOps\docker-compose.yml down -v --remove-orphans
docker compose -f .\MLOps\docker-compose.yml up -d --build
```

Acessos:

```text
Airflow:   http://localhost:8080   airflow / airflow
Streamlit: http://localhost:8501
MinIO:     http://localhost:9001   minioadmin / minioadmin123
```
