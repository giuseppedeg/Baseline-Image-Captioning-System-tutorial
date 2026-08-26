# Chapter 0 — Overview

## What this tutorial builds

A system that takes an image and produces a sentence describing it. The tutorial builds three such systems, in increasing order of capability, and evaluates all three with the same instruments so that the differences between them are measurable rather than asserted.

| Stage | Encoder | Decoder | Training |
|-------|---------|---------|----------|
| 1 | Pre-trained, frozen | Written from scratch (LSTM, then Transformer) | Decoder only |
| 2 | Pre-trained, progressively unfrozen | As above, conditioned on auxiliary predictions | Decoder, auxiliary heads, then encoder |
| 3 | Pre-trained vision-language model | The model's own decoder | Full or parameter-efficient fine-tuning |

Stages 1 and 2 are implemented in plain PyTorch, deliberately. Every mechanism that matters (how a caption becomes a sequence of integers, how the decoder is prevented from reading its own future, how a beam is maintained during search) is visible in code that the reader can step through. Stage 3 then uses the Hugging Face ecosystem, and the contrast is instructive in both directions: what the library saves, and what it hides.


## Prerequisites

The reader is assumed to be comfortable with Python, to have trained a classifier in PyTorch at some point, and to know what a convolution and a softmax are. No prior exposure to sequence modelling, attention or vision-language models is assumed; each is introduced where it is first needed.

The tutorial assumes access to a single modern GPU (I think you have acces to the CVC clusters, right??). Stage 1 trains in minutes on a corpus of a few thousand images; Stage 3 is the only stage where memory becomes a real constraint, and the corresponding chapter discusses what to do
with less of it.

## Scope, stated honestly

The system produces descriptions that are **plausible and stylistically coherent**. Given a photograph of a building it will characterise its type, its materials, its apparent style and its approximate period. It will not tell you which building it is, who built it, or in what year — and Chapter 1 argues that this is a property of the problem rather than a deficiency of the model.

This matters for how the output should be used. A caption produced by this system is a description, not a record. If your application needs the latter, the appendix on retrieval-augmented captioning describes the architecture that provides it, and explains why it is a different system rather than a better version of this one.


## Installation

```bash
git clone https://github.com/giuseppedeg/Baseline-Image-Captioning-System-tutorial
cd Baseline-Image-Captioning-System-tutorial

python -m venv .venv && source .venv/bin/activate   # or conda, or uv
pip install -r requirements.txt
pip install -e .                                     # makes `import captioning` work

# Optional but recommended: the named-entity recogniser used in Chapter 1.
# A rule-based fallback runs without it.
python -m spacy download en_core_web_trf
```

## Expected input

Two things: a directory of images, and a CSV with one row per image.

```
data/
├── data.csv
└── img/
    ├── 001.jpg
    ├── 002.jpg
    └── ...
```

```csv
ID,Path,name,century,caption
001,img/001.jpg,TajMahal,17,"Built in the 17th century by Mughal Emperor Shah Jahan ..."
```

| column | role |
|--------|------|
| `ID` | unique row identifier |
| `Path` | image location, relative to `data.image_root` |
| `name` | subject identifier — read but **not** used as a training target |
| `century` | integer century; a configurable sentinel marks "unknown" |
| `caption` | free-text description |

Two properties of this schema drive most of the design decisions that follow, and both are examined in Chapter 1: captions are unique per image, and `name` is therefore close to a unique key rather than a class label.

If your corpus uses different headers, map them in the configuration file rather than rewriting the data:

```yaml
data:
  columns:
    image_id: image_uuid
    image_path: filename
    caption: description
```

The corpus is assumed to be partitioned into splits already. Point `data.train_csv`, `data.val_csv` and `data.test_csv` at the corresponding files.


## The two preprocessing steps

Both are run once, before any training.

```bash
# Derive grounded captions. Adds caption_raw and caption_grounded columns.
python scripts/prepare_captions.py \
    --input-csv data/train.csv --output-csv data/train_processed.csv --report

# Fit the tokeniser on the training split only.
python scripts/build_tokenizer.py --config configs/stage1_scratch.yaml
```

Read the report printed by the first command before continuing. It states what
was removed from the captions and why, and it is the cheapest opportunity in
the whole pipeline to discover that the preprocessing is deleting the content
you meant to keep.

## Repository layout

```
configs/                 one YAML file per stage
src/captioning/          the library: importable, testable, free of side effects
  data/                  corpus access, tokenisation, caption grounding
  models/                encoders, decoders, auxiliary heads
  training/              losses, training loop, schedules, checkpoints
  inference/             decoding strategies, attention extraction
  evaluation/            metrics and reporting
  utils/                 configuration, logging, reproducibility
scripts/                 entry points: parse arguments, call the library
docs/                    this tutorial
tests/                   unit tests
```

The boundary between `src/` and `scripts/` is enforced rather than decorative.
Nothing under `src/` reads command-line arguments, writes to the filesystem outside of an explicit `save` call, or prints. Everything under `scripts/` is short enough to read in one sitting. Research code organised this way remains usable after the experiment it was written for.

## Conventions used throughout

**Configuration over flags.** A run is described by a YAML file that can be versioned alongside its results. Command-line arguments exist only for overrides during exploration. Unknown keys in a configuration file raise an error instead of being ignored — a setting one believes is applied but is not is among the more expensive bugs in experimental work.

**Masks mark what is forbidden.** In `padding_mask` and in the causal mask, `True` means *this position must not be attended to*, following the convention of `torch.nn.MultiheadAttention`.

**Fit on training data only.** Vocabularies, normalisation statistics and any other quantity estimated from data are estimated from the training split. This is stated here once and assumed everywhere after.

## Reading order

1. [Problem formulation](01_problem_formulation.md) — what a captioning model
   can and cannot learn, and what that implies for the data.
2. [Stage 1](02_stage1_encoder_decoder.md) — a frozen encoder and a decoder
   trained from scratch.
3. [Stage 2](02_stage1_encoder_decoder.md) — auxiliary supervision and conditioned generation.
4. Stage 3 — fine-tuning a pre-trained vision-language model.
5. Appendix — retrieval-augmented captioning.

Chapter 1 is short and contains no code you need to run. It is also the chapter that determines whether the rest of the tutorial makes sense.
