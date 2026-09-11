# Training data

Both datasets are public and verified under `leonli66`:

| Dataset | Hugging Face repository | Pinned revision |
|---|---|---|
| Raw | [stage3-final-mixture-20260910-raw](https://huggingface.co/datasets/leonli66/stage3-final-mixture-20260910-raw) | `f72b2fafa7dec00c03b272b24bc83c4c27db6bc3` |
| Packed | [stage3-final-mixture-20260910-packed](https://huggingface.co/datasets/leonli66/stage3-final-mixture-20260910-packed) | `f46f4124d6c602ec74d931d6999def02b6d9e7bb` |

Anonymous access and every inventory file's size/content hash passed verification.
See [the verification record](data-verification.json). Each repository includes
`release.json`, `UPLOAD_STATUS.json`, original build manifests, audit records,
source revisions and upstream notices. No `state.json` or uploader cache is included.
Upstream terms remain applicable.

## Download packed data

Choose **one** length; the two versions contain overlapping source examples.

```python
from pathlib import Path
from huggingface_hub import snapshot_download

version = "packed-cs16-32768"  # Or "packed-cs16-16384".
root = snapshot_download(
    repo_id="leonli66/stage3-final-mixture-20260910-packed",
    repo_type="dataset",
    revision="f46f4124d6c602ec74d931d6999def02b6d9e7bb",
    allow_patterns=[f"{version}/**", "release.json", "UPLOAD_STATUS.json"],
)
dataset_root = str(Path(root) / version)
```

Point `stages[3].dataset` at `dataset_root`, which contains `data/mixed/` and
`manifest.json`. Set the matching `stages[3].max_packed_length` and keep
`training.compression_ratio: 16`, `data.packed_attention_backend: flash`, and
loader file/row shuffling enabled. Payloads use LCLM's LargeBinary Arrow format
and require `DynamicPackedDataset`.

| Category | 16,384-token sequences | 32,768-token sequences |
|---|---:|---:|
| Agent | 273,564 | 159,428 |
| Reasoning | 712,993 | 465,966 |
| Other | 1,292,364 | 651,616 |
| **Total** | **2,278,921** | **1,277,010** |
| Original examples retained | 21,463,686 | 21,669,429 |
| Parquet files | 4,452 | 2,495 |

Every sequence contains one category. Completed sequences were globally
permuted using PCG64, seed 20260910. Native and expanded agents occur throughout
the final shards. All permutation positions and serialized payloads were checked;
all eight sampled ranks receive every category and both agent kinds.

Limits count decoder positions **after 16:1 compression**, including memory
boundary tags. Decoder IDs and assistant-only labels are precomputed; memory
strings are encoder-tokenized at runtime. Memory vectors and boundary tags have
label `-100`. Native agents stay uncompressed. Reasoning follows the corrected
CoT50 policy with prompts intact. Expanded-agent observations use memory segments.

Whole examples exceeding a version's compressed length limit were excluded, not
truncated. Of the 152,503 expanded examples, 16k retains 152,498 and 32k retains
all 152,503. The raw dataset preserves examples excluded from either packed version.

## Raw examples

Raw means processed training examples before tokenization/packing, rather than
upstream downloads or generation logs.

| HF configuration | Rows |
|---|---:|
| `base` | 20,326,114 |
| `native_agents` | 1,244,170 |
| `expanded_agents` | 152,503 |
| **Total** | **21,722,787** |

```python
from datasets import load_dataset

agents = load_dataset(
    "leonli66/stage3-final-mixture-20260910-raw",
    "expanded_agents",
    revision="f72b2fafa7dec00c03b272b24bc83c4c27db6bc3",
    split="train",
    streaming=True,
)
```

Native/expanded `messages` and `tools` fields are lossless JSON strings; decode
with `json.loads` before applying a chat template. Native agents originate from
OpenThoughts-Agent-SFT-100K, Nemotron-Agentic-v1 and Nemotron-SFT-Agentic-v2.
The expanded union contains reviewed retained/pilot traces and new Qwen
non-thinking traces. Rejected and held pools are excluded. Exact source and
teacher lineage is in each repository's `provenance/` directory.

## Reproducibility

Pinned tokenizers, also defined in `data/stage3_tokenizers.py`:

- Decoder: `Qwen/Qwen3-4B-Instruct-2507`,
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Encoder: `Qwen/Qwen3-Embedding-0.6B`,
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

Packed manifest SHA256 values:

- 16k: `02461223c31dc5958cc742afefdd72881428253ca90af0e91d103477717a9681`
- 32k: `38bbd44bd600e1c6fdcc2c4ad89cba1297ead9c156b28e03f1fad9cb0982650f`

Existing Modal copies are on `lclm-stage3-data` at
`/data/stage3-build-20260906/global-shuffle-20260910-v2/packed-cs16-{16384,32768}`.

Reusable tokenization, category packing and shuffle functions remain in
`data/preprocess_for_dynamic_packing.py`, `data/grouped_stage3_packing.py` and
`data/global_sequence_shuffle.py`, with regression tests. The completed build's
one-off generation, recovery and publication scripts are preserved in
[Git history at c943335](https://github.com/LeonLixyz/LCLM/tree/c94333562bf8d7cc8162cc970b2519b20b1cf253).
Training from the published snapshots does not require them.
