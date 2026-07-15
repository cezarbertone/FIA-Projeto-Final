"""Camada de I/O abstrata: filesystem local OU MinIO.

Todos os scripts usam caminhos lógicos iguais (ex.: ``Dados/abt.csv``). Quando
``STORAGE_BACKEND=minio``, esses caminhos são object keys dentro do bucket. Assim,
o pipeline não produz localmente para depois sincronizar: ele lê e grava diretamente
no data lake, como no projeto em grupo.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
MIRROR_LOCAL_OUTPUTS = os.getenv("MIRROR_LOCAL_OUTPUTS", "true").lower() == "true"

# Mesmo usando MinIO como data lake/fonte de verdade, estes artefatos oficiais
# também são replicados no repositório para facilitar avaliação e demonstração.
LOCAL_MIRROR_PATHS = {
    "Dados/raw_data.csv",
    "Dados/clean_data.csv",
    "Dados/abt.csv",
    "Model/model.pkl",
    "Model/metrics.json",
    "Model/reference_profile.json",
}
_s3 = None
_bucket_ready = False


def _use_minio() -> bool:
    return STORAGE_BACKEND == "minio"


def _bucket() -> str:
    value = os.getenv("MINIO_BUCKET", "home-credit-empresarios")
    if not value:
        raise RuntimeError("MINIO_BUCKET não configurado.")
    return value


def _client():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
        )
    return _s3


def _key(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def local_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_bucket() -> None:
    """Garante a existência do bucket de forma idempotente.

    O ``minio-init`` pode criar o bucket ao mesmo tempo em que uma task do
    Airflow começa a gravar um artefato. Nesse intervalo, o ``list_buckets``
    ainda pode não refletir o bucket e duas chamadas podem tentar criá-lo.
    O MinIO responde ``BucketAlreadyOwnedByYou``; esse retorno significa que
    o estado desejado já foi atingido e não deve derrubar a task.
    """
    global _bucket_ready

    if not _use_minio() or _bucket_ready:
        return

    client = _client()
    bucket = _bucket()
    names = {item["Name"] for item in client.list_buckets().get("Buckets", [])}

    if bucket not in names:
        try:
            client.create_bucket(Bucket=bucket)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", ""))

            # Condição de corrida segura: outro serviço/processo criou o bucket
            # entre o list_buckets e o create_bucket.
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise

    _bucket_ready = True


def exists(path: str | Path) -> bool:
    if not _use_minio():
        return local_path(path).exists()
    try:
        _client().head_object(Bucket=_bucket(), Key=_key(path))
        return True
    except Exception:
        return False



def _should_mirror(path: str | Path) -> bool:
    return MIRROR_LOCAL_OUTPUTS and _key(path) in LOCAL_MIRROR_PATHS


def _mirror_file(src: str | Path, path: str | Path) -> None:
    """Replica localmente os artefatos oficiais, sem mudar o MinIO como backend principal."""
    if not _should_mirror(path):
        return
    dst = local_path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    if src_path.resolve() != dst.resolve():
        shutil.copy2(src_path, dst)


def _mirror_bytes(payload: bytes, path: str | Path) -> None:
    if not _should_mirror(path):
        return
    dst = local_path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)


def upload_file(local_file: str | Path, path: str | Path) -> None:
    src = Path(local_file)
    if _use_minio():
        ensure_bucket()
        _client().upload_file(str(src), _bucket(), _key(path))
        _mirror_file(src, path)
    else:
        dst = local_path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def download_file(path: str | Path, local_file: str | Path) -> Path:
    dst = Path(local_file)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _use_minio():
        _client().download_file(_bucket(), _key(path), str(dst))
    else:
        src = local_path(path)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    return dst


def materialize(path: str | Path) -> Path:
    """Disponibiliza um objeto como arquivo local para leitura em chunks."""
    if not _use_minio():
        return local_path(path)
    key = _key(path)
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache = Path(tempfile.gettempdir()) / "home_credit_storage" / f"{digest}_{Path(key).name}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Baixa novamente em cada task/run para evitar cache obsoleto.
    _client().download_file(_bucket(), key, str(cache))
    return cache


def _write_temp_and_upload(writer, path: str | Path, suffix: str) -> None:
    if not _use_minio():
        dst = local_path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        writer(dst)
        return
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        writer(tmp_path)
        upload_file(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(materialize(path), **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> None:
    _write_temp_and_upload(lambda p: df.to_csv(p, index=index, **kwargs), path, ".csv")


def read_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(materialize(path), **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> None:
    _write_temp_and_upload(lambda p: df.to_parquet(p, index=index, **kwargs), path, ".parquet")


def read_json(path: str | Path):
    if _use_minio():
        obj = _client().get_object(Bucket=_bucket(), Key=_key(path))
        return json.loads(obj["Body"].read().decode("utf-8"))
    with open(local_path(path), encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str | Path, **kwargs) -> None:
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("ensure_ascii", False)
    payload = json.dumps(obj, **kwargs).encode("utf-8")
    if _use_minio():
        ensure_bucket()
        _client().put_object(Bucket=_bucket(), Key=_key(path), Body=payload)
        _mirror_bytes(payload, path)
    else:
        dst = local_path(path); dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(payload)


def read_pickle(path: str | Path):
    if _use_minio():
        obj = _client().get_object(Bucket=_bucket(), Key=_key(path))
        return pickle.loads(obj["Body"].read())
    with open(local_path(path), "rb") as f:
        return pickle.load(f)


def write_pickle(obj, path: str | Path) -> None:
    payload = pickle.dumps(obj)
    if _use_minio():
        ensure_bucket(); _client().put_object(Bucket=_bucket(), Key=_key(path), Body=payload)
        _mirror_bytes(payload, path)
    else:
        dst = local_path(path); dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(payload)


def list_keys(prefix: str = "") -> list[str]:
    if not _use_minio():
        base = local_path(prefix)
        if base.is_file(): return [_key(prefix)]
        return [_key(p.relative_to(PROJECT_ROOT)) for p in base.rglob("*") if p.is_file()] if base.exists() else []
    keys, token = [], None
    while True:
        args = {"Bucket": _bucket(), "Prefix": _key(prefix)}
        if token: args["ContinuationToken"] = token
        result = _client().list_objects_v2(**args)
        keys.extend(item["Key"] for item in result.get("Contents", []))
        if not result.get("IsTruncated"): break
        token = result["NextContinuationToken"]
    return keys
