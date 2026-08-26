# A Baseline Image Captioning System — A Step-by-Step Tutorial

This repository  walks through the construction of an image captioning system from first principles, starting from a frozen visual encoder with a decoder trained from scratch and ending with a fine-tuned vision-language model (..for now, it's just a plan, i didnt prepare the fining tune of the vision language model). 
Each stage is a working system, evaluated with the same metrics, so that the effect of every architectural decision can be measured rather than assumed.

The tutorial want to explain *why* a captioning architecture looks the way it does...

## What this system does, and what it does not

Given an image, the system produces a description that is **plausible and stylistically coherent**: it characterises what kind of object or structure is depicted, its apparent style and materials, and its approximate historical period.

It does **not** identify the specific subject of the photograph, and it does not recover facts that are not visible in the image (who commissioned a building, the name of the person it commemorates, the exact year of its completion...). 
This boundary is not a limitation to be apologised for; it follows directly from what the input signal contains, and Chapter 1 makes the argument precisely. 



## The three stages

| Stage | System | Concepts introduced |
|-------|--------|---------------------|
| **1** | Frozen encoder + decoder trained from scratch (LSTM, then Transformer), implemented in plain PyTorch | Tokenisation, teacher forcing, autoregressive decoding, beam search, cross-attention and attention maps |
| **2** | Multi-task model: optional auxiliary heads for period and typology, with the decoder conditioned on their predictions; progressive unfreezing of the encoder (this is where the rest of the CSV metadata is used, not only the description... not yet implemented, but easy to do) | Auxiliary supervision, ordinal targets, loss weighting, conditioned generation, transfer schedules |
| **3** | Fine-tuning a pre-trained vision-language model (BLIP / GIT) with the Hugging Face ecosystem (an idea for a later step; not implemented) | Transfer learning at scale, parameter-efficient fine-tuning, an honest comparison against Stage 1 |

Everything is deliberately written in plain PyTorch. It is a good way to understand
the classical structure of a PyTorch project. Stage 3 could be written using some
more advanced libraries.

## Status

| | state |
|---|---|
| Preprocessing, data package, tokenisers | implemented, tested |
| Stage 1 — frozen encoder, decoder from scratch | implemented, tested |
| Evaluation suite and shared results table | implemented, tested |
| Stage 2 — auxiliary heads, progressive unfreezing | designed, not built |
| Stage 3 — vision-language fine-tuning | an idea, not planned in detail |
| Appendix — retrieval-augmented captioning | an idea, not planned in detail |

Chapters 0 to 2 of the guide are written. `pytest tests/` covers the data package,
the model and mask semantics, the search strategies and the metrics.

The auxiliary heads of Stage 2 are designed as optional components enabled from
configuration, because whether a given corpus supports them is an empirical
question rather than an assumption — see §1.6 of the guide.


## Repository layout

```
configs/                 YAML configuration, one file per stage
src/captioning/          The library. Pure functions and classes, no side effects.
  data/                  Datasets, tokenisers, transforms, entity masking
  models/                Encoders, decoders, auxiliary heads, composition
  training/              Losses, training engine, schedules, checkpointing
  inference/             Decoding strategies, attention extraction
  evaluation/            Text metrics, grounding metrics, factual metrics, reporting
  utils/                 Configuration, logging, seeding
scripts/                 Thin entry points: parse arguments, call the library
docs/                    The tutorial itself, one chapter per stage
tests/                   Unit tests for the library
```

The separation is a deliberate part of the repository. Everything in `src/` is importable and testable in isolation; everything in `scripts/` is a few dozen lines that read a configuration and delegate. Research code that keeps this boundary stays usable after the paper is submitted.

## Expected data format

>!NOTE
> I have based everithing on the database i have defined (it's a toy dataset!!).
> It is different from yours ( i cannot remember the exact format you showd me)
> but everything can be easily adapted.


A CSV with one row per image, alongside a directory of images:

| column | meaning |
|--------|---------|
| `ID` | unique identifier |
| `Path` | image path, relative to `data.image_root` |
| `name` | subject identifier (not used as a training target — see Chapter 1) |
| `century` | integer century, with a configurable sentinel for unknown |
| `caption` | free-text description |

Column names are mapped in `configs/*.yaml` under `data.columns`, so a corpus using different headers needs no modification of the data files themselves. The corpus is assumed to be already partitioned into splits. A five-row sample lives under [`data/`](data/) so that the pipeline can be run before a real corpus is available; read [`data/README.md`](data/README.md) first, as its licensing is unresolved.


## Getting started

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf     # optional; a rule-based fallback exists

# 1. Derive grounded captions from the raw ones
python scripts/prepare_captions.py --config configs/stage1_scratch.yaml \
    --input-csv data/train.csv --output-csv data/train_processed.csv --report

# 2. Fit the tokeniser on the training split
python scripts/build_tokenizer.py --config configs/stage1_scratch.yaml

# 3. Train and evaluate Stage 1
python scripts/train_stage1.py --config configs/stage1_scratch.yaml
python scripts/evaluate.py --config configs/stage1_scratch.yaml \
    --checkpoint runs/stage1/best.pt --split val --name stage1-transformer
```

Then read [`docs/00_overview.md`](docs/00_overview.md) and work through the
chapters in order.

## Reading order

0. [Overview and setup](docs/00_overview.md)
1. [Problem formulation: what a caption model can learn](docs/01_problem_formulation.md)
2. [Stage 1: a decoder trained from scratch](docs/02_stage1_encoder_decoder.md)

Subsequent chapters accompany Stages 2 and 3.

## License

MIT, see [LICENSE](LICENSE). The sample data under `data/` is **not** covered by
it and is not cleared for redistribution; see [`data/README.md`](data/README.md).
