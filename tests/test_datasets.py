"""Tests for the WikiText-2 loader (Hub access mocked, parquet real)."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from silicon_eval.evals import datasets
from silicon_eval.exceptions import DatasetLoadError

EXPECTED_NAME = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def write_parquet(path: Path, rows: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"text": rows}), path)


class ForbiddenApi:
    """The cache-friendly path must never list the repo."""

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        raise AssertionError("list_repo_files must not be called when the known name resolves")


def test_load_wikitext2_uses_known_name_without_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = tmp_path / "test.parquet"
    write_parquet(parquet, [" = Heading = \n", "\n", "Body line.\n"])

    def fake_download(repo_id: str, filename: str, *, repo_type: str) -> str:
        assert filename == EXPECTED_NAME
        return str(parquet)

    monkeypatch.setattr(datasets, "HfApi", ForbiddenApi)
    monkeypatch.setattr(datasets, "hf_hub_download", fake_download)

    assert datasets.load_wikitext2_text() == " = Heading = \n\nBody line.\n"


def test_falls_back_to_repo_listing_when_layout_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = tmp_path / "renamed.parquet"
    write_parquet(parquet, ["Body line.\n"])
    renamed = "wikitext-2-raw-v1/test-00000-of-00002.parquet"

    def fake_download(repo_id: str, filename: str, *, repo_type: str) -> str:
        if filename == EXPECTED_NAME:
            raise EntryNotFoundError("gone")
        assert filename == renamed
        return str(parquet)

    class FakeApi:
        def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
            assert repo_id == "Salesforce/wikitext"
            assert repo_type == "dataset"
            return ["wikitext-2-raw-v1/train-00000-of-00001.parquet", renamed]

    monkeypatch.setattr(datasets, "HfApi", FakeApi)
    monkeypatch.setattr(datasets, "hf_hub_download", fake_download)

    assert datasets.load_wikitext2_text() == "Body line.\n"


def test_download_failure_wrapped_in_dataset_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline_download(repo_id: str, filename: str, *, repo_type: str) -> str:
        raise ConnectionError("no network")

    monkeypatch.setattr(datasets, "hf_hub_download", offline_download)
    with pytest.raises(DatasetLoadError, match="could not load WikiText-2 'test'"):
        datasets.load_wikitext2_text()


def test_missing_split_raises_dataset_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(repo_id: str, filename: str, *, repo_type: str) -> str:
        raise EntryNotFoundError("gone")

    class EmptyApi:
        def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
            return ["wikitext-2-raw-v1/train-00000-of-00001.parquet"]

    monkeypatch.setattr(datasets, "HfApi", EmptyApi)
    monkeypatch.setattr(datasets, "hf_hub_download", not_found)
    with pytest.raises(DatasetLoadError, match="no 'test' parquet"):
        datasets.load_wikitext2_text()
