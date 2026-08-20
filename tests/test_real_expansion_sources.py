from data.real_expansion_sources import (
    EVAL_DATASET_BLACKLIST,
    EVAL_SOURCE_BLACKLIST,
    SOURCE_SPECS,
)


def test_registry_uses_only_training_partitions():
    for source in SOURCE_SPECS:
        assert all(partition.split == "train" for partition in source.partitions)


def test_registry_does_not_import_evaluation_datasets():
    registered_ids = {source.source_id.lower() for source in SOURCE_SPECS}
    for dataset_id in EVAL_DATASET_BLACKLIST:
        assert dataset_id.lower() not in registered_ids


def test_longbench_sources_are_explicitly_blacklisted():
    required = {
        "qasper",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "repobench-p",
    }
    assert required <= set(EVAL_SOURCE_BLACKLIST)


def test_requested_domains_and_sources_are_present():
    keys = {source.key for source in SOURCE_SPECS}
    assert {"multidoc2dial", "techqa", "finqa", "maud", "cuad", "acord"} <= keys
    assert {
        "dialogue",
        "support",
        "finance",
        "legal",
        "code",
    } <= {source.category for source in SOURCE_SPECS}


def test_artificial_pubmedqa_is_not_registered():
    pubmed = next(source for source in SOURCE_SPECS if source.key == "pubmedqa_labeled")
    assert {partition.config for partition in pubmed.partitions} == {"pqa_labeled"}


def test_natural_questions_is_gated_as_a_large_source():
    source = next(source for source in SOURCE_SPECS if source.key == "natural_questions")
    assert source.status == "large_source"
    assert source.train_eligible


def test_derived_sources_declare_lineage():
    for source in SOURCE_SPECS:
        if source.status == "derived_duplicate":
            assert source.parent_sources


def test_contract_nli_is_automatically_harvestable():
    contract_nli = next(source for source in SOURCE_SPECS if source.key == "contract_nli")
    assert contract_nli.acquisition == "github"
    assert contract_nli.license == "CC-BY-4.0"
