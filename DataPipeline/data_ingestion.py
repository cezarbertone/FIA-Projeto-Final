"""Ingestão da Landing para a camada oficial raw_data.csv e fontes auxiliares.

A Landing fica fora de Dados. A task apenas promove os arquivos recebidos; nenhuma
limpeza ou regra de negócio é aplicada nesta etapa.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import pandas as pd
from DataPipeline.config import (
    LANDING_DIR, RAW_DATA_PATH, RAW_BUREAU_PATH, RAW_PREVIOUS_PATH, RAW_INSTALLMENTS_PATH,
    MAX_ROWS_APPLICATION, MAX_ROWS_BUREAU, MAX_ROWS_PREVIOUS, MAX_ROWS_INSTALLMENTS,
    find_landing_file,
)
from MLOps import storage

SOURCES = {
    "application_train": (RAW_DATA_PATH, MAX_ROWS_APPLICATION),
    "bureau": (RAW_BUREAU_PATH, MAX_ROWS_BUREAU),
    "previous_application": (RAW_PREVIOUS_PATH, MAX_ROWS_PREVIOUS),
    "installments_payments": (RAW_INSTALLMENTS_PATH, MAX_ROWS_INSTALLMENTS),
}


def _promote(source: Path, destination: str, max_rows: int) -> dict:
    if max_rows > 0:
        df = pd.read_csv(source, nrows=max_rows, low_memory=False)
        storage.write_csv(df, destination)
        rows = len(df)
    else:
        storage.upload_file(source, destination)
        rows = None
    return {
        "source": str(source), "destination": destination, "bytes": source.stat().st_size,
        "debug_rows": rows,
    }


def run() -> dict:
    storage.ensure_bucket()
    items = {}
    for logical, (destination, limit) in SOURCES.items():
        items[logical] = _promote(find_landing_file(logical), destination, limit)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "landing_dir": str(LANDING_DIR), "storage_backend": storage.STORAGE_BACKEND,
        "files": items,
    }
    storage.write_json(report, "reports/ingestion_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

if __name__ == "__main__": run()
