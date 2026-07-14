"""Eval corpus loaders (Hugging Face Hub, cached locally by huggingface_hub)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from silicon_eval.exceptions import DatasetLoadError

_WIKITEXT_REPO = "Salesforce/wikitext"
_WIKITEXT_SUBSET = "wikitext-2-raw-v1"


def load_wikitext2_text(split: str = "test") -> str:
    """Load a WikiText-2 (raw) split as one concatenated string.

    Rows in the source parquet are lines that already carry their newlines,
    so joining with the empty string reconstructs the original text.
    Raises :class:`DatasetLoadError` on any download or parse failure.
    """
    try:
        local_path = _download_split(split)
        return _read_text_column(Path(local_path))
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(f"could not load WikiText-2 {split!r} split: {exc}") from exc


def _download_split(split: str) -> str:
    """Fetch the split's parquet, preferring the stable well-known filename.

    ``hf_hub_download`` serves a known filename from the local cache even when
    offline; listing the repo needs a live connection, so it is only a
    fallback for the day the upstream shard layout changes.
    """
    expected = f"{_WIKITEXT_SUBSET}/{split}-00000-of-00001.parquet"
    try:
        return hf_hub_download(_WIKITEXT_REPO, expected, repo_type="dataset")
    except EntryNotFoundError:
        return hf_hub_download(_WIKITEXT_REPO, _find_parquet_file(split), repo_type="dataset")


def _find_parquet_file(split: str) -> str:
    files = HfApi().list_repo_files(_WIKITEXT_REPO, repo_type="dataset")
    prefix = f"{_WIKITEXT_SUBSET}/{split}-"
    for name in files:
        if name.startswith(prefix) and name.endswith(".parquet"):
            return name
    raise DatasetLoadError(
        f"no {split!r} parquet found under {_WIKITEXT_SUBSET} in {_WIKITEXT_REPO}"
    )


def _read_text_column(path: Path) -> str:
    import pyarrow.parquet as pq  # deferred: pyarrow import is slow

    table = pq.read_table(path, columns=["text"])
    rows = cast("list[str]", table.column("text").to_pylist())
    return "".join(rows)
