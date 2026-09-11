import copy
import random
import weakref
from collections import Counter

import pytest

from data.expansion_source_probes import (
    LEXGLUE_CONFIGS, classify_source_prefix, select_balanced_probe, select_source_probe,
)
from data.synthetic_expansion_agent import TASK_FAMILIES


def task(config, index):
    return {
        "task_id": f"original-{config}-{index}",
        "source_row_id": f"{config}-{index}",
        "family": "lex_glue",
        "_attempt_provenance": {"kind": "previously_unattempted"},
        "segments": [{"segment_id": "seg_1", "text": "original source text"}],
    }


def rows(capacity=20):
    return [task(config, index) for config in LEXGLUE_CONFIGS for index in range(capacity)]


def test_probe_covers_all_configs_and_preserves_exact_original_objects():
    original = rows()
    before = copy.deepcopy(original)
    selected, capacities, quotas = select_source_probe(iter(original))
    assert len(selected) == len({row["task_id"] for row in selected}) == 64
    assert capacities == dict.fromkeys(LEXGLUE_CONFIGS, 20)
    assert quotas == {config: 10 if config == "case_hold" else 9 for config in LEXGLUE_CONFIGS}
    assert Counter(classify_source_prefix(row) for row in selected) == quotas
    assert {classify_source_prefix(row) for row in selected[:7]} == set(LEXGLUE_CONFIGS)
    by_id = {row["task_id"]: row for row in original}
    assert all(row is by_id[row["task_id"]] for row in selected)
    assert original == before


def test_selection_and_order_ignore_stream_order_but_use_seed():
    original = rows(40)
    shuffled = original.copy()
    random.Random(1234).shuffle(shuffled)
    first = select_source_probe(iter(original), seed="probe-A")
    assert first == select_source_probe(iter(reversed(original)), seed="probe-A")
    assert first == select_source_probe(iter(shuffled), seed="probe-A")
    assert first[0] != select_source_probe(iter(original), seed="probe-B")[0]


def test_quotas_adjust_to_capacities_and_report_absent_configs():
    original = [task("case_hold", 0)] + [task("scotus", i) for i in range(100)]
    selected, capacities, quotas = select_source_probe(iter(original))
    assert len(selected) == 64
    assert quotas == {config: {"case_hold": 1, "scotus": 63}.get(config, 0)
                      for config in LEXGLUE_CONFIGS}
    assert capacities == {config: {"case_hold": 1, "scotus": 100}.get(config, 0)
                          for config in LEXGLUE_CONFIGS}


def test_capacity_shortage_fails_instead_of_returning_partial_probe():
    with pytest.raises(ValueError, match="exceeds available"):
        select_source_probe(iter(rows(9)))
    with pytest.raises(ValueError, match="exceeds available"):
        select_source_probe(iter([]))


@pytest.mark.parametrize("identifier", [
    None, 2, "", "case_hold", "case_hold-", "case_hold--1", "case_hold-01",
    "case_hold-1.5", "case_hold-2-extra", "case_hold-1\n", "unknown-1", "SCOTUS-1",
])
def test_unknown_or_malformed_config_rejected(identifier):
    row = task("case_hold", 0)
    row["source_row_id"] = identifier
    with pytest.raises(ValueError, match="source_row_id"):
        select_source_probe([row], total=1)


def test_duplicate_unselected_id_fails_even_across_configs():
    original = rows(30)
    selected_ids = {row["task_id"] for row in select_source_probe(original)[0]}
    unselected = next(row for row in original if row["task_id"] not in selected_ids)
    duplicate = task("unfair_tos", 1000)
    duplicate["task_id"] = unselected["task_id"]
    with pytest.raises(ValueError, match="Duplicate probe task_id"):
        select_source_probe(iter([*original, duplicate]))


@pytest.mark.parametrize("identifier", [None, 1, "", " padded "])
def test_malformed_task_ids_rejected(identifier):
    row = task("scotus", 0)
    row["task_id"] = identifier
    with pytest.raises(ValueError, match="task_id"):
        select_source_probe([row], total=1)


@pytest.mark.parametrize("total", [-1, True, 1.5, "64"])
def test_invalid_budget_rejected(total):
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_source_probe([], total=total)


def test_zero_budget_still_validates_source_and_reports_capacity():
    selected, capacities, quotas = select_source_probe([task("scotus", 0)], total=0)
    assert selected == []
    assert capacities["scotus"] == 1 and sum(capacities.values()) == 1
    assert quotas == dict.fromkeys(LEXGLUE_CONFIGS, 0)
    with pytest.raises(ValueError, match="Duplicate"):
        select_source_probe([task("scotus", 0), task("scotus", 0)], total=0)


def test_synthetic_uses_known_family_without_inventing_source_ids():
    original = [{"task_id": f"native-{family}-{i}", "family": family}
                for family in TASK_FAMILIES for i in range(20)]
    selected, capacities, quotas = select_source_probe(iter(original), source="synthetic")
    assert len(selected) == 64
    assert set(quotas) == set(TASK_FAMILIES)
    assert min(quotas.values()) == 12 and max(quotas.values()) == 13
    assert capacities == dict.fromkeys(TASK_FAMILIES, 20)
    assert all("source_row_id" not in row for row in selected)
    with pytest.raises(ValueError, match="Unknown synthetic"):
        select_source_probe([{"task_id": "bad", "family": "invented"}], 1, source="synthetic")
    with pytest.raises(ValueError, match="Unsupported probe source"):
        select_source_probe([], source="invented")


def test_generic_selector_consumes_once_and_does_not_retain_all_payloads():
    class TrackedTask(dict):
        __hash__ = object.__hash__

    live = weakref.WeakSet()
    peak = 0

    def stream():
        nonlocal peak
        for index in range(1000):
            row = TrackedTask(task_id=f"task-{index}", group="a" if index % 2 else "b")
            live.add(row)
            peak = max(peak, len(live))
            yield row

    selected, capacities, quotas = select_balanced_probe(
        stream(), 10, classify=lambda row: row["group"], strata=("a", "b"),
    )
    assert len(selected) == 10 and capacities == {"a": 500, "b": 500}
    assert quotas == {"a": 5, "b": 5}
    # Two strata times ten candidates, plus current iterator/local references.
    assert peak <= 24
    assert len(live) == 10


def test_generic_classifier_cannot_silently_add_unknown_strata():
    with pytest.raises(ValueError, match="Unknown probe stratum"):
        select_balanced_probe([{"task_id": "x"}], 1,
                              classify=lambda row: "unknown", strata=("a",))
