# Final Stage-3 Hugging Face publication

Both datasets are **uploaded and verified**, with public visibility under
`leonli66`. The initial upload completed at 05:24 UTC on September 11, 2026.
Public access and updated metadata were verified at 06:54 UTC (02:54 Eastern). These are new repositories for the final reviewed build:

- [Raw examples](https://huggingface.co/datasets/leonli66/stage3-final-mixture-20260910-raw)
- [Packed 16k and 32k](https://huggingface.co/datasets/leonli66/stage3-final-mixture-20260910-packed)

## Pinned revisions and verification

| Dataset | Final revision including completion marker | Verified files | Parquet files | Verified bytes including metadata |
|---|---|---:|---:|---:|
| Raw | `f72b2fafa7dec00c03b272b24bc83c4c27db6bc3` | 2,766 | 2,733 | 180,014,880,061 |
| Packed | `f46f4124d6c602ec74d931d6999def02b6d9e7bb` | 6,980 | 6,947 | 352,312,448,546 |

Every inventory-listed file is present and matches its frozen source size and
content hash. LFS SHA256 hashes were checked against the build inventory;
non-LFS metadata was checked against its local SHA256 and remote Git blob hash.
No unexpected files were found other than the generated `.gitattributes` and
`UPLOAD_STATUS.json`. No `state.json` or uploader caches are included.
Both repositories' completion markers say `complete`.

Final public verification: [machine-readable report](data/reviews/stage3-final-hf-publication-20260911.json).
Anonymous access was checked for both repositories. Only dataset cards, release
metadata and completion markers changed during publication; all data and build
audit files remain identical. Local upload history and original private-snapshot
verification are preserved under `_modal_run/hf-publication-20260911/`.

## Raw data

There are **21,722,787 processed training examples before tokenization/packing**:

| HF configuration | Rows | Files |
|---|---:|---|
| `base` | 20,326,114 | `raw/base/*.parquet` |
| `native_agents` | 1,244,170 | `raw/native_agents/*.parquet` |
| `expanded_agents` | 152,503 | `raw/expanded_agents/*.parquet` |

Generation failures/debug outputs are excluded. Native/expanded `messages` and
`tools` are lossless JSON strings; apply `json.loads` before using a chat template.
For example:

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

## Packed data for training

| Version root | Sequences | Original examples retained | Parquet files |
|---|---:|---:|---:|
| `packed-cs16-16384` | 2,278,921 | 21,463,686 | 4,452 |
| `packed-cs16-32768` | 1,277,010 | 21,669,429 | 2,495 |

Select **one** version; their underlying examples overlap. Each version root
contains its original `manifest.json` and `data/mixed/` shards. Download a pinned
snapshot and pass the selected version root to `DynamicPackedDataset`:

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

Set `stages[3].dataset` to `dataset_root` and `stages[3].max_packed_length` to
16384 or 32768 accordingly. Compression ratio is 16. Lengths include memory
boundary tags after compression. Decoder IDs and assistant-only labels are
precomputed; memory text is encoder-tokenized at runtime. Sequences contain one
category (agent, reasoning or other), then are globally shuffled across categories.
See [packing details](STAGE3_PACKED_DATA_20260910.md) for counts, source policy,
tokenizer pins and manifest hashes, and [training instructions](AGENTS.md) to launch.

Both repositories contain dataset cards, exact inventories, pinned provenance,
preserved upstream notices and relevant audit records. Public availability does
not replace upstream terms or expand the recorded scope of source reviews. The upload used existing Modal
volume artifacts and CPU workers; no generation, repacking or production
training was launched. Multi-node/RDMA and full model/optimizer checkpoint
restore remain outside the completed training validation scope.
