from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
LANDING_DIR = PROJECT_ROOT / "Landing"  # fora de Dados, conforme arquitetura oficial

# Caminhos LÓGICOS. No backend local são arquivos; no MinIO são object keys.
DATA_DIR = "Dados"
RAW_DATA_PATH = "Dados/raw_data.csv"
CLEAN_DATA_PATH = "Dados/clean_data.csv"
ABT_PATH = "Dados/abt.csv"
PROCESSING_DIR = "Dados/_processing"

RAW_BUREAU_PATH = f"{PROCESSING_DIR}/raw_bureau.csv"
RAW_PREVIOUS_PATH = f"{PROCESSING_DIR}/raw_previous_application.csv"
RAW_INSTALLMENTS_PATH = f"{PROCESSING_DIR}/raw_installments_payments.csv"
CLEAN_BUREAU_PATH = f"{PROCESSING_DIR}/clean_bureau.csv"
CLEAN_PREVIOUS_PATH = f"{PROCESSING_DIR}/clean_previous_application.csv"
FEATURES_BUREAU_PATH = f"{PROCESSING_DIR}/features_bureau.csv"
FEATURES_PREVIOUS_PATH = f"{PROCESSING_DIR}/features_previous_application.csv"
FEATURES_INSTALLMENTS_PATH = f"{PROCESSING_DIR}/features_installments_payments.csv"

REPORTS_DIR = "reports"
MODEL_PATH = "Model/model.pkl"
METRICS_PATH = "Model/metrics.json"
REFERENCE_PROFILE_PATH = "Model/reference_profile.json"

RAW_FILES = {
    "application_train": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
}

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
SMALL_BUSINESS_COL = "SMALL_BUSINESS_PROXY"
SMALL_BUSINESS_ORGANIZATION_TYPES = {"Self-employed"}
SMALL_BUSINESS_INCOME_TYPES = {"Commercial associate", "Businessman"}
FILTER_SMALL_BUSINESS = os.getenv("FILTER_SMALL_BUSINESS", "true").lower() == "true"

DAYS_EMPLOYED_ANOMALY = 365243
NULL_DROP_THRESHOLD = float(os.getenv("NULL_DROP_THRESHOLD", "0.50"))
PROTECTED_NULL_COLUMNS = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.50"))

# Limites opcionais de debug; 0 = base completa.
MAX_ROWS_APPLICATION = int(os.getenv("MAX_ROWS_APPLICATION", "0") or 0)
MAX_ROWS_BUREAU = int(os.getenv("MAX_ROWS_BUREAU", "0") or 0)
MAX_ROWS_PREVIOUS = int(os.getenv("MAX_ROWS_PREVIOUS", "0") or 0)
MAX_ROWS_INSTALLMENTS = int(os.getenv("MAX_ROWS_INSTALLMENTS", "0") or 0)
INSTALLMENTS_CHUNKSIZE = int(os.getenv("INSTALLMENTS_CHUNKSIZE", "500000"))

# Treinamento no padrão do projeto em grupo.
TEST_SIZE = float(os.getenv("MODEL_TEST_SIZE", "0.20"))
CV_FOLDS = int(os.getenv("MODEL_CV_FOLDS", "5"))
SEARCH_SAMPLE_FRAC = float(os.getenv("MODEL_SEARCH_SAMPLE_FRAC", "0.30"))
ANALYSIS_THRESHOLDS = [
    float(x.strip()) for x in os.getenv("MODEL_ANALYSIS_THRESHOLDS", "0.30,0.50,0.70").split(",") if x.strip()
]
PERMUTATION_SAMPLE_ROWS = int(os.getenv("PERMUTATION_SAMPLE_ROWS", "5000"))
PERMUTATION_REPEATS = int(os.getenv("PERMUTATION_REPEATS", "3"))
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "true").lower() == "true"
SHAP_SAMPLE_ROWS = int(os.getenv("SHAP_SAMPLE_ROWS", "500"))

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "home-credit-empresarios")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()


def find_landing_file(logical_name: str) -> Path:
    base = RAW_FILES[logical_name].replace(".csv", "")
    candidates: list[Path] = []
    for pattern in (f"{base}.csv", f"{base}*.csv", f"{base}.CSV", f"{base}*.CSV"):
        candidates.extend(sorted(LANDING_DIR.glob(pattern)))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"{RAW_FILES[logical_name]} não encontrado em {LANDING_DIR}. "
        "Extraia arquivos compactados e coloque os CSVs na pasta Landing da raiz."
    )
