# SEFRQO-Plus: A Self-Evolving Query Optimizer via Large Language Models and Retrieval-Augmented Generation

The VLDB Journal, 2026. Paper: https://doi.org/10.1007/s00778-026-00999-z

SEFRQO-Plus builds a feedback-oriented vector database with query structure-aware embeddings and multi-layered histories as references, and drives prompt optimization with multi-dimensional historical feedback for continuous self-evolution. It trains the LLM with a customized query group relative policy optimization. This repository contains the online optimizer, one configuration per experiment setting, and the GRPO training code with its JOB training data.

## Requirements

- Python 3.10
- PostgreSQL with [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan) in `shared_preload_libraries`
- NVIDIA GPU

## Install

```bash
conda create -n sefrqo-plus python=3.10
conda activate sefrqo-plus
pip install -r requirements.txt
```

Database connections are set in `sefrqo_plus/config.py`.

```bash
python load_postgres/prepend_imdb_headers.py --csv_dir <imdb csv dir>
bash load_postgres/load_job_postgres.sh <imdb csv dir> imdbload 5438
```

Download the checkpoints from https://www.dropbox.com/scl/fo/y08hmd9s1q2dc34fwuqmv/AAgTB0TIfKMDpIdBo39Ve-w?rlkey=1sku3fckkvzegezpavhc5c2gu&st=lf18cpm4&e=1&dl=0 and place them under `models/`:

```text
models/sft-3b                 3B-JOB-ckpts/ckpt-4296
models/sft-8b                 8B-JOB-Apr2-ckpts/ckpt-1074
models/sft-grpo-3b            Apr12-3B-GRPO/checkpoint-500
models/sft-grpo-8b            Apr13-8B-GRPO/checkpoint-100
models/sft-grpo-3b-dynamic    Latest-3B-GRPO/checkpoint-100
```

## Run

```bash
python -m sefrqo_plus configs/plus_static_ceb_lsg8b.json
```

Runs SEFRQO-Plus with the 8B SFT and GRPO model on CEB. Each file in `configs/` is one setting of the paper.

```bash
python -m training.grpo.train_grpo --model <SFT checkpoint>
```

Fine-tunes an SFT model with GRPO.

## Repository layout

```text
SEFRQO-PLUS/
├── sefrqo_plus/     Online optimizer
│   ├── __main__.py  Entry point
│   ├── online.py    Main loop
│   ├── prompts.py   Prompt construction
│   ├── records.py   Execution records
│   ├── encoders.py  Query encoders
│   ├── models.py    Model loading
│   ├── postgres.py  PostgreSQL access
│   └── config.py    Database connection
├── configs/         Experiment settings
├── data/
│   ├── prompts/     System prompts
│   └── history/     Reference queries
├── sql/             Workloads
├── training/
│   └── grpo/        GRPO training
└── load_postgres/   Database loading
```

## Citation

```bibtex
@article{liu2026sefrqoplus,
  author  = {Hanwen Liu and Qihan Zhang and Ryan Marcus and Ibrahim Sabek},
  title   = {{SEFRQO-Plus}: A Self-Evolving Query Optimizer via Large Language Models and Retrieval-Augmented Generation},
  journal = {The {VLDB} Journal},
  volume  = {35},
  number  = {6},
  pages   = {58},
  year    = {2026},
  doi     = {10.1007/s00778-026-00999-z}
}
```

## License

GPL-3.0-or-later. See `LICENSE`.
