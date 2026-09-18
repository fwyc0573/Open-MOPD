"""Full-train parquet must not go through unbounded Dataset.from_list."""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from verl.utils.dataset.rl_dataset import _load_parquet_dataset


FULL_TRAIN = Path("/data/ycfeng/tmp/mopd-step1-open-mopd-train.parquet")
EXPECTED_COUNTS = {"math": 17917, "code": 23667, "if": 45347}


def test_nested_parquet_loads_without_from_list(tmp_path) -> None:
    table = pa.table(
        {
            "prompt": pa.array([[{"role": "user", "content": "q"}]], type=pa.list_(pa.struct(
                [("role", pa.string()), ("content", pa.string())]
            ))),
            "domain": ["math"],
            "reward_model": pa.array(
                [{"style": "rule", "ground_truth": "1"}],
                type=pa.struct([("style", pa.string()), ("ground_truth", pa.string())]),
            ),
        }
    )
    path = tmp_path / "nested.parquet"
    pq.write_table(table, path)
    dataset = _load_parquet_dataset(str(path))
    assert len(dataset) == 1
    assert dataset[0]["domain"] == "math"


def test_full_train_row_counts_if_local_file_exists() -> None:
    if os.environ.get("MOPD_LOAD_FULL_TRAIN") != "1" or not FULL_TRAIN.is_file():
        return
    dataset = _load_parquet_dataset(str(FULL_TRAIN))
    assert len(dataset) == sum(EXPECTED_COUNTS.values())
    domains = dataset["domain"]
    counts = {name: sum(1 for item in domains if str(item) == name) for name in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS
