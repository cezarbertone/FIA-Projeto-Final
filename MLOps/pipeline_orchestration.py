from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_inputs():
    from Tools.check_inputs import run
    return run()


def ingest():
    from DataPipeline.data_ingestion import run
    return run()


def clean():
    from DataPipeline.data_sanitization import run
    return run()


def aggregate():
    from DataPipeline.feature_aggregation import run
    return run()


def build_abt():
    from DataPipeline.abt_transform import run
    return run()


def train():
    from Model.train import train
    return train()


def score():
    from Model.predict import score_sample
    return score_sample(2000).shape[0]


def run_all():
    steps = [check_inputs, ingest, clean, aggregate, build_abt, train, score]
    for fn in steps:
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    run_all()
