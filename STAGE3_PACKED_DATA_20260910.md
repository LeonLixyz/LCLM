# Stage-3 packs — September 10, 2026

Both versions are packed and globally shuffled, including native agents and the
reviewed expansion traces. Final loader and independent source/rank audits
passed for both lengths at 19:36 UTC (15:36 Eastern). The data is ready for the
validated LCLM FSDP or DeepSpeed ZeRO-2 setup described below.

Both versions are also uploaded and verified in the public Hugging Face dataset
[`leonli66/stage3-final-mixture-20260910-packed`](https://huggingface.co/datasets/leonli66/stage3-final-mixture-20260910-packed),
revision `f46f4124d6c602ec74d931d6999def02b6d9e7bb`. Select the
`packed-cs16-16384` or `packed-cs16-32768` subdirectory as the training root.
The [HF publication handoff](HF_PUBLICATION_20260911.md) includes a pinned
download recipe and the corresponding raw dataset.

| Category | 16,384-token sequences | 32,768-token sequences |
|---|---:|---:|
| agent | 273,564 | 159,428 |
| reasoning | 712,993 | 465,966 |
| other | 1,292,364 | 651,616 |
| **Total sequences** | **2,278,921** | **1,277,010** |
| **Original examples retained** | **21,463,686** | **21,669,429** |

Every sequence contains one category: agent, reasoning or other. All completed
sequences were then globally permuted together using PCG64, seed 20260910. Native
and expanded agents are distributed throughout the final shards. Every original
sequence, example and serialized payload was preserved; all shuffle positions
were checked, and all final files were reopened and verified.

On Modal volume `lclm-stage3-data`, mounted at `/data`, use these version roots:

- `/data/stage3-build-20260906/global-shuffle-20260910-v2/packed-cs16-16384`
- `/data/stage3-build-20260906/global-shuffle-20260910-v2/packed-cs16-32768`

These supersede the earlier `grouped-packing-20260909-v3` training paths. The
original packs remain available as provenance. The globally shuffled versions
contain 4,452 and 2,495 Parquet files, respectively, all under `data/mixed/`.
Point `DynamicPackedDataset` at the version root with `shuffle=True`,
`shuffle_files=True`, `compression_ratio=16`, and `target_length=16384` or `32768`.
Use the corresponding version alone; the 16k/32k variants contain overlapping
source examples and should not be concatenated as distinct data.

Length limits count decoder tokens **after 16:1 compression**, including memory
boundary tags. Decoder IDs and assistant-only labels are preprocessed. Memory
text is stored separately and encoder-tokenized at training time; memory tokens
and their boundary tags do not receive prediction loss. Native agent traces stay
uncompressed. Reasoning uses the corrected CoT50 policy, keeping prompts intact.

Pinned tokenizers:

- Decoder: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Encoder: `Qwen/Qwen3-Embedding-0.6B`, revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

The reviewed expansion union contains 152,503 examples: 70,343 corrected LexGLUE,
22,627 current main-stream/probe examples, and 59,533 retained legacy/pilot
examples after common filters and global-ID checks. At 16k, five whole examples
exceed the compressed limit, leaving 152,498; 32k retains all 152,503. No expansion
examples were truncated. These contribute 23,919 agent sequences at 16k and
11,703 at 32k; native agents contribute 249,645 and 147,725, respectively.
Previously held source/configuration pools remain outside the reviewed selection.
All 70,343 selected corrected-Lex traces passed the tokenizer/native/label audit.

The actual loader passed on final LargeBinary shards at both lengths, covering
native agents, expanded agents, reasoning and other data. It verified padded
shapes, exact compressed lengths, supervised-token counts and memory-label masks.
The independent audit bound all 4,493 original 16k files and 2,535 original 32k
files to their source certificates/expansion audits. Every rank's first 4,096
sequences contains native agents, expansion agents, reasoning and other data.
With eight ranks and the default drop policy, the loader assigns equal lengths
and omits only the epoch remainder: one sequence at 16k or two at 32k. These
sequences remain in the pack files.

GPU verification passed 44 regression tests, 8-rank DDP/FSDP stress tests, and
real 16k/32k training updates with the pretrained 4B decoder and 0.6B encoder on
one H200:8 FSDP node. This applies to the tested LCLM stage-3 build clone. The
separate Code-LLaVA implementation and multi-node training are not cleared by
these results. The follow-up DeepSpeed optimizer fix passed full-model ZeRO-2 at
both lengths. ZeRO-2 is the intended setup and maximum supported ZeRO stage;
the unnecessary ZeRO-3 investigation and experimental changes were dropped.
See [training verification](TRAINING_READINESS_20260910.md) and the
[DeepSpeed report](DEEPSPEED_ORDERING_20260910.md).

Final globally shuffled manifest SHA256 values:

- 16k: `02461223c31dc5958cc742afefdd72881428253ca90af0e91d103477717a9681`
- 32k: `38bbd44bd600e1c6fdcc2c4ad89cba1297ead9c156b28e03f1fad9cb0982650f`
- Completion: `84da5a08fd3b25e5c2d3d6b79c714802bc8643fb4c185b8ab555b78d5f78cb09`
- Independent audit: `c1937b150760babc3e26b023e3e3c89b384f48feaad0a22a90266815dfd1a357`

Exact final manifests and audit records are saved locally under
`_modal_run/global-shuffle-final-20260910/`; loader and GPU reports are in
`_modal_run/training-readiness-final/`. Original build evidence remains in
`_modal_run/final-packing-20260910/`. Public Hugging Face access was verified
on September 11; no production training run was launched.
