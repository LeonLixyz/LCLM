"""Deterministic source probes with bounded payload memory and exact ID checks.

No models, network access, or Modal calls. A temporary on-disk ID index makes
duplicate detection exact without retaining every source task or ID in RAM.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.expansion_retry_diagnostic import fair_quotas
from data.synthetic_expansion_agent import TASK_FAMILIES


LEXGLUE_CONFIGS = (
    "case_hold", "ecthr_a", "ecthr_b", "eurlex", "ledgar", "scotus", "unfair_tos",
)
DEFAULT_PROBE_SEED = "expansion-source-probe-v1"
Task = Mapping[str, Any]


def classify_source_prefix(task: Task, *, source: str = "lex_glue") -> str:
    """Return a validated LexGLUE config or procedural synthetic family.

    LexGLUE IDs are generated as ``config-<zero-based row number>``. Synthetic
    tasks identify their stratum by ``family``, not a fabricated source-row ID.
    """
    if not isinstance(task, Mapping):
        raise ValueError("Probe task must be a mapping")
    if source == "lex_glue":
        identifier = task.get("source_row_id")
        if not isinstance(identifier, str):
            raise ValueError("LexGLUE source_row_id must be a string")
        match = re.fullmatch(r"([a-z_]+)-(0|[1-9][0-9]*)", identifier)
        if match is None or match[1] not in LEXGLUE_CONFIGS:
            raise ValueError("Unknown or malformed LexGLUE source_row_id: " + identifier)
        return match[1]
    if source == "synthetic":
        family = task.get("family")
        if not isinstance(family, str) or family not in TASK_FAMILIES:
            raise ValueError("Unknown synthetic task family: " + str(family))
        return family
    raise ValueError("Unsupported probe source: " + str(source))


@dataclass
class _Candidate:
    key: tuple[bytes, str]
    task: Task

    def __lt__(self, other: _Candidate) -> bool:
        # Reverse ordering: heap root is the worst retained stable-hash rank.
        return self.key > other.key


def select_balanced_probe(
    tasks: Iterable[Task],
    total: int = 64,
    *,
    classify: Callable[[Task], str],
    strata: Sequence[str],
    seed: str = DEFAULT_PROBE_SEED,
) -> tuple[list[Task], dict[str, int], dict[str, int]]:
    """Balance available strata and return original tasks, capacities, quotas.

    Consumes the input exactly once; keeps at most ``total`` task objects per
    stratum. Exact duplicate checking uses a temporary SQLite index with a 1 MiB
    page-cache target. Disk use is proportional to IDs, never task payloads.

    Selection and output order are independent of input order. Quotas are equal
    where capacity permits, with deterministic alphabetical tie-breaking. Output
    interleaves strata so the first round covers each selected stratum. The task
    objects and all their provenance are preserved unchanged.
    """
    if type(total) is not int or total < 0:
        raise ValueError("Probe total must be a nonnegative integer")
    if not isinstance(seed, str):
        raise ValueError("Probe seed must be a string")
    names = tuple(strata)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Probe strata must be nonempty, distinct strings")
    names = tuple(sorted(names))
    capacities = dict.fromkeys(names, 0)
    heaps: dict[str, list[_Candidate]] = {name: [] for name in names}

    with tempfile.TemporaryDirectory(prefix="expansion-probe-ids-") as directory:
        connection = sqlite3.connect(Path(directory) / "ids.sqlite")
        try:
            connection.execute("PRAGMA cache_size=-1024")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
            for task in tasks:
                if not isinstance(task, Mapping):
                    raise ValueError("Probe task must be a mapping")
                task_id = task.get("task_id")
                if not isinstance(task_id, str) or not task_id or task_id.strip() != task_id:
                    raise ValueError("Probe task_id must be a nonempty trimmed string")
                try:
                    connection.execute("INSERT INTO ids VALUES (?)", (task_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate probe task_id: " + task_id) from exc
                name = classify(task)
                if not isinstance(name, str) or name not in capacities:
                    raise ValueError("Unknown probe stratum: " + str(name))
                capacities[name] += 1
                if total == 0:
                    continue
                identity = json.dumps([seed, name, task_id], ensure_ascii=False,
                                      separators=(",", ":")).encode()
                candidate = _Candidate((hashlib.sha256(identity).digest(), task_id), task)
                heap = heaps[name]
                if len(heap) < total:
                    heapq.heappush(heap, candidate)
                elif candidate.key < heap[0].key:
                    heapq.heapreplace(heap, candidate)
        finally:
            connection.close()

    if total > sum(capacities.values()):
        raise ValueError("Requested probe exceeds available tasks")
    quotas = dict.fromkeys(names, 0)
    quotas.update(fair_quotas(capacities, total))
    ranked = {name: sorted(heaps[name], key=lambda item: item.key)[:quotas[name]]
              for name in names}
    selected = [ranked[name][rank].task
                for rank in range(max(quotas.values(), default=0))
                for name in names if rank < quotas[name]]
    return selected, capacities, quotas


def select_source_probe(
    tasks: Iterable[Task],
    total: int = 64,
    *,
    source: str = "lex_glue",
    seed: str = DEFAULT_PROBE_SEED,
) -> tuple[list[Task], dict[str, int], dict[str, int]]:
    """Select a balanced LexGLUE-config or synthetic-family source probe."""
    if source == "lex_glue":
        strata = LEXGLUE_CONFIGS
    elif source == "synthetic":
        strata = TASK_FAMILIES
    else:
        raise ValueError("Unsupported probe source: " + str(source))
    return select_balanced_probe(
        tasks, total, classify=lambda task: classify_source_prefix(task, source=source),
        strata=strata, seed=seed,
    )
