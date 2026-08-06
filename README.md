# DocICL

Implementation of *DocICL: Document-Level In-Context Learning for Event Causality
Identification with Large Language Models* (anonymous ACL submission).

One LLM call labels every event mention pair of a document, using few-shot examples that
are whole annotated documents carrying a synthetic chain of thought, with predictions
aggregated by per-pair vote over several resamplings.

## Install

```bash
uv sync
export OPENROUTER_API_KEY=<key>
```

## Run

Set `active_dataset` in `config.yaml` to `event_story_line`, `causal_timebank` or
`maven_ere`, then:

```bash
uv run main.py
```

Micro precision, recall and F1 are printed at the end of the run and written, with the
per-pair predictions and a copy of the configuration, to `logs/allatonce/`.

The three corpora are included under `data/`. Cross-validation follows the paper: 10 folds
for Causal-TimeBank, 5 for EventStoryLine, none for MAVEN-ERE. Few-shot examples are drawn
only from the training folds.

## Configuration

`config.yaml` holds every parameter. The main configuration of the paper is
`deepseek-ai/DeepSeek-V4-Pro` as backbone (`model.default_model_id`), three examples per
prompt (`few_shot.n_examples`) drawn at random (`few_shot.selection`) from a pool of thirty
annotated documents (`few_shot.pool_size`), and three resamplings aggregated by majority
vote (`experiment.resampling.n_runs`, with `enabled` and `distinct_fewshots` both true).
The chains of thought come from the three-step protocol under `few_shot.cot_generation`,
which needs `enabled`, `blind` and `rewrite` set.

Chains of thought are synthesized once per corpus, then cached in `logs/cot_cache.json` and
reused across documents and resamplings. Delete that file to regenerate them. Inference
itself is never cached.

## Prompts

`prompts/` holds one system prompt and one user template per corpus, selected by
`datasets.<key>.prompt`. Their content is reproduced in the appendix of the paper.
