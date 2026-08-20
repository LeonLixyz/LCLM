"""Registry of real-document sources for selective-expansion training.

The registry separates raw acquisition from source-specific conversion.  A
source may be mirrored even when it is not yet eligible for the final training
mixture; licensing, lineage deduplication, and evaluation holdouts are enforced
again during conversion and publication.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


SourceCategory = Literal[
    "dialogue", "support", "finance", "legal", "code", "science", "government", "web"
]
SourceStatus = Literal[
    "ready",
    "documents_only",
    "large_source",
    "license_review",
    "derived_duplicate",
    "identifier_needed",
]
AcquisitionMode = Literal["huggingface", "github", "external"]


@dataclass(frozen=True)
class Partition:
    config: str
    split: str = "train"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    category: SourceCategory
    adapter: str
    acquisition: AcquisitionMode
    source_id: str
    license: str
    partitions: tuple[Partition, ...] = ()
    status: SourceStatus = "ready"
    parent_sources: tuple[str, ...] = ()
    notes: str = ""

    @property
    def train_eligible(self) -> bool:
        return self.status in {"ready", "documents_only", "large_source"}

    def as_manifest_record(self) -> dict[str, object]:
        record = asdict(self)
        record["train_eligible"] = self.train_eligible
        return record


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    # Human document-grounded dialogue and support.
    SourceSpec(
        key="multidoc2dial",
        category="dialogue",
        adapter="multidoc2dial",
        acquisition="huggingface",
        source_id="IBM/multidoc2dial",
        license="CC-BY-3.0",
        partitions=(
            Partition("document_domain"),
            Partition("dialogue_domain"),
            Partition("multidoc2dial"),
        ),
        notes="Human multi-document government-service dialogues with grounding spans.",
    ),
    SourceSpec(
        key="doc2dial",
        category="dialogue",
        adapter="doc2dial",
        acquisition="huggingface",
        source_id="IBM/doc2dial",
        license="CC-BY-3.0",
        partitions=(Partition("document_domain"), Partition("dialogue_domain")),
        status="derived_duplicate",
        parent_sources=("multidoc2dial",),
        notes="Original corpus is contained in MultiDoc2Dial; do not duplicate its documents or dialogues.",
    ),
    SourceSpec(
        key="faithdial",
        category="dialogue",
        adapter="faithdial",
        acquisition="huggingface",
        source_id="McGill-NLP/FaithDial",
        license="MIT",
        partitions=(Partition("plain_text"),),
        notes="Human-edited knowledge-grounded dialogue with evidence passages.",
    ),
    SourceSpec(
        key="techqa",
        category="support",
        adapter="techqa",
        acquisition="huggingface",
        source_id="PrimeQA/TechQA",
        license="Apache-2.0",
        partitions=(Partition("default"),),
        notes="Real developer-forum questions linked to IBM Technotes.",
    ),
    SourceSpec(
        key="watsonx_docs_qa",
        category="support",
        adapter="watsonx_docs_qa",
        acquisition="huggingface",
        source_id="ibm-research/watsonxDocsQA",
        license="Apache-2.0",
        partitions=(Partition("corpus"), Partition("question_answers")),
        notes="Document corpus plus grounded support questions and document IDs.",
    ),
    SourceSpec(
        key="clapnq",
        category="support",
        adapter="clapnq",
        acquisition="huggingface",
        source_id="PrimeQA/clapnq",
        license="Apache-2.0",
        partitions=(Partition("default"),),
        notes="Human questions with contextualized long answers grounded in Wikipedia documents.",
    ),
    # Finance: preserve tables, prose, evidence, and executable programs.
    SourceSpec(
        key="finqa",
        category="finance",
        adapter="finqa",
        acquisition="huggingface",
        source_id="bevaya/FinQA",
        license="MIT + CDLA-Permissive-1.0 source tables",
        partitions=(Partition("default"),),
        notes="Full-fidelity FinQA train data with evidence and reasoning programs.",
    ),
    SourceSpec(
        key="tatqa",
        category="finance",
        adapter="tatqa",
        acquisition="huggingface",
        source_id="FangyuLei/tatqa",
        license="BSD-3-Clause",
        partitions=(Partition("default"),),
        notes="Financial tables and text with span/arithmetic answers.",
    ),
    SourceSpec(
        key="multihiertt",
        category="finance",
        adapter="multihiertt",
        acquisition="huggingface",
        source_id="bevaya/MultiHiertt",
        license="MIT",
        partitions=(Partition("default"),),
        notes="Multi-table hierarchical financial reasoning with evidence labels.",
    ),
    SourceSpec(
        key="convfinqa",
        category="finance",
        adapter="convfinqa",
        acquisition="github",
        source_id="https://github.com/czyssrs/ConvFinQA.git",
        license="MIT",
        notes="Human conversational financial QA with turn-level programs.",
    ),
    SourceSpec(
        key="financebench",
        category="finance",
        adapter="financebench",
        acquisition="huggingface",
        source_id="PatronusAI/financebench",
        license="CC-BY-NC-4.0",
        partitions=(Partition("default"),),
        status="license_review",
        notes="Grounded finance QA; non-commercial restriction requires explicit mixture approval.",
    ),
    SourceSpec(
        key="fa_v2",
        category="finance",
        adapter="fa_v2",
        acquisition="external",
        source_id="FAv2",
        license="unknown",
        status="identifier_needed",
        notes="User-requested source; exact public repository/dataset identifier still needed.",
    ),
    # Expert-annotated legal corpora.
    SourceSpec(
        key="maud",
        category="legal",
        adapter="maud",
        acquisition="huggingface",
        source_id="theatticusproject/maud",
        license="CC-BY-4.0",
        partitions=(Partition("default"),),
        notes="Expert-labeled merger-agreement questions over 92 deal points.",
    ),
    SourceSpec(
        key="cuad",
        category="legal",
        adapter="cuad",
        acquisition="huggingface",
        source_id="theatticusproject/cuad",
        license="CC-BY-4.0",
        partitions=(Partition("default"),),
        notes="Full commercial contracts with expert clause annotations.",
    ),
    SourceSpec(
        key="cuad_qa",
        category="legal",
        adapter="cuad_qa",
        acquisition="huggingface",
        source_id="theatticusproject/cuad-qa",
        license="CC-BY-4.0",
        partitions=(Partition("default"),),
        status="derived_duplicate",
        parent_sources=("cuad",),
        notes="QA rendering of CUAD; use for prompts but deduplicate against CUAD documents.",
    ),
    SourceSpec(
        key="acord",
        category="legal",
        adapter="acord",
        acquisition="huggingface",
        source_id="theatticusproject/acord",
        license="CC-BY-4.0",
        partitions=(Partition("default"),),
        notes="Expert-rated query-clause retrieval pairs.",
    ),
    SourceSpec(
        key="contract_nli",
        category="legal",
        adapter="contract_nli",
        acquisition="github",
        source_id="https://github.com/stanfordnlp/contract-nli.git",
        license="CC-BY-4.0",
        notes="Full NDAs with entailment, contradiction, and evidence-span labels.",
    ),
    SourceSpec(
        key="legalbench",
        category="legal",
        adapter="legalbench",
        acquisition="huggingface",
        source_id="nguha/legalbench",
        license="CC-BY-4.0",
        status="derived_duplicate",
        parent_sources=("maud", "cuad", "contract_nli"),
        notes="Use train partitions and unique task families only; many tasks derive from parent corpora.",
    ),
    SourceSpec(
        key="legalbench_rag",
        category="legal",
        adapter="legalbench_rag",
        acquisition="github",
        source_id="https://github.com/ZeroEntropy-AI/legalbenchrag.git",
        license="mixed upstream licenses",
        status="documents_only",
        parent_sources=("maud", "cuad", "contract_nli"),
        notes="Harvest corpus and character-span mappings; do not train on benchmark test queries.",
    ),
    SourceSpec(
        key="lex_glue",
        category="legal",
        adapter="lex_glue",
        acquisition="huggingface",
        source_id="coastalcph/lex_glue",
        license="CC-BY-4.0",
        partitions=tuple(
            Partition(config)
            for config in (
                "case_hold",
                "ecthr_a",
                "ecthr_b",
                "eurlex",
                "ledgar",
                "scotus",
                "unfair_tos",
            )
        ),
        notes="Real cases, legislation, and contracts; classification labels need natural task rendering.",
    ),
    # Real code issues. Repository content is fetched later at the pinned base commit.
    SourceSpec(
        key="swe_gym",
        category="code",
        adapter="swe_issue",
        acquisition="huggingface",
        source_id="SWE-Gym/SWE-Gym",
        license="MIT metadata; preserve per-repository licenses",
        partitions=(Partition("default"),),
        notes="Real GitHub issues and patches; exclude every official SWE-bench evaluation instance.",
    ),
    SourceSpec(
        key="swe_bench_extra",
        category="code",
        adapter="swe_issue",
        acquisition="huggingface",
        source_id="nebius/SWE-bench-extra",
        license="CC-BY-4.0 metadata; preserve per-repository licenses",
        partitions=(Partition("default"),),
        notes="Additional real issue/patch pairs outside the official SWE-bench set.",
    ),
    # Additional real documents that support grounded expansion traces.
    SourceSpec(
        key="pubmedqa_labeled",
        category="science",
        adapter="pubmedqa",
        acquisition="huggingface",
        source_id="qiaojin/PubMedQA",
        license="MIT",
        partitions=(Partition("pqa_labeled"),),
        notes="Expert-labeled biomedical questions grounded in PubMed abstracts; exclude artificial split.",
    ),
    SourceSpec(
        key="billsum",
        category="government",
        adapter="billsum",
        acquisition="huggingface",
        source_id="FiscalNote/billsum",
        license="CC0-1.0",
        partitions=(Partition("default"),),
        notes="US legislative text with human-authored summaries; use train split only.",
    ),
    SourceSpec(
        key="natural_questions",
        category="web",
        adapter="natural_questions",
        acquisition="huggingface",
        source_id="google-research-datasets/natural_questions",
        license="CC-BY-SA-3.0",
        partitions=(Partition("default"),),
        status="large_source",
        notes="Large human web-query corpus with Wikipedia documents; harvest separately due size.",
    ),
)


EVAL_DATASET_BLACKLIST: tuple[str, ...] = (
    "tonychenxyz/ruler-full",
    "nimitkalra/LongBench-v1",
    "THUDM/LongBench",
    "tonychenxyz/longhealth",
    "leonli66/longhealth5",
    "tonychenxyz/codellava-gsm8k-plain",
)

EVAL_SOURCE_BLACKLIST: tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
    "dureader",
    "vcsum",
    "lsht",
)


def validate_registry() -> None:
    keys = [spec.key for spec in SOURCE_SPECS]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate real-expansion source key")
    known = set(keys)
    for spec in SOURCE_SPECS:
        missing = set(spec.parent_sources) - known
        if missing:
            raise ValueError(f"{spec.key}: unknown parent sources {sorted(missing)}")
        if spec.acquisition == "huggingface" and "/" not in spec.source_id:
            raise ValueError(f"{spec.key}: invalid Hugging Face dataset ID")
        if spec.status == "ready" and not spec.partitions and spec.acquisition == "huggingface":
            raise ValueError(f"{spec.key}: ready Hugging Face source has no partitions")


validate_registry()
