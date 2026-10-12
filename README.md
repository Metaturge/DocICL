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

## Causal event retrieval

`rankers.py` ranks, for each event mention annotated as an effect, the other mentions of its
document as candidate causes, from the vote counts of finished runs, and compares them with
TF-IDF cosine, embedding cosine, word distance and random rankers:

```bash
uv run rankers.py <run> [<run> ...]
```

`<run>` is a report path or its 8-hex run id in `logs/allatonce/`. MRR, Hit@k, Recall@k and
nDCG@k are printed for every sentence scope (all, intra, inter), directed and undirected, and
written to `<report>.ranking.json`. Settings are under `ranking` in `config.yaml`. The
similarity rankers need `scikit-learn` and `sentence-transformers`.

## Prompts

`prompts/` holds one system prompt and one user template per corpus, selected by
`datasets.<key>.prompt`. Their content is reproduced in the appendix of the paper.
EventStoryLine and MAVEN-ERE also have a prompt that covers inter-sentence relations, used
for causal event retrieval: `*_lean_coref`, which also asks the model to group the mentions
of one event and to link every mention of a cause to every mention of its effect.

## Further analyses

Results that are not in the paper. Bold marks the best score in each column, italics the
second best.

### Few-shot example selection

DeepSeek-V4-Flash, $k=3$ examples per prompt and $N=3$ resamplings. The examples are drawn at
random (the main configuration, from a pool of 30 annotated documents) or ranked by
similarity to the target document, from a pool of 100: TF-IDF cosine, sentence embedding
cosine (`all-roberta-large-v1`) or Jaccard similarity of the mention sets.

| Selection | ESC P | ESC R | ESC F1 | CTB P | CTB R | CTB F1 | MAVEN-ERE P | MAVEN-ERE R | MAVEN-ERE F1 |
|---|---|---|---|---|---|---|---|---|---|
| random (main) | **37.3** | 70.1 | **48.7** | **57.5** | *57.7* | *57.6* | 38.8 | **60.9** | 47.4 |
| TF-IDF | 31.8 | **77.4** | 45.1 | 52.2 | 56.7 | 54.3 | **41.1** | 55.7 | 47.3 |
| sentence embedding | *35.8* | 72.2 | *47.9* | 55.7 | 57.4 | 56.5 | *41.0* | 58.3 | **48.1** |
| mention-set Jaccard | 34.7 | *77.2* | *47.9* | *56.8* | **60.1** | **58.4** | 40.3 | *58.5* | *47.7* |

### Vote threshold

DeepSeek-V4-Flash with $N=7$ resamplings. A pair is predicted causal when at least $t$
resamplings predict it, from the union of the resamplings ($t=1$) to their unanimity
($t=7$). (\*) The majority vote.

| $t$ | ESC P | ESC R | ESC F1 | CTB P | CTB R | CTB F1 | MAVEN-ERE P | MAVEN-ERE R | MAVEN-ERE F1 |
|---|---|---|---|---|---|---|---|---|---|
| ≥ 1 (union) | 24.5 | **94.5** | 38.9 | 36.0 | **81.9** | 50.1 | 28.0 | **84.0** | 42.0 |
| ≥ 2 | 29.4 | *89.1* | 44.2 | 45.3 | *70.8* | 55.2 | 33.6 | *74.5* | 46.3 |
| ≥ 3 | 34.2 | 83.0 | 48.4 | 52.0 | 65.8 | *58.1* | 37.5 | 66.5 | *48.0* |
| ≥ 4 (majority)\* | 40.0 | 74.0 | *51.9* | 57.2 | 59.7 | **58.5** | 40.9 | 58.6 | **48.2** |
| ≥ 5 | 44.3 | 65.6 | **52.9** | 63.6 | 52.7 | 57.6 | 44.7 | 50.5 | 47.4 |
| ≥ 6 | *50.1* | 53.0 | 51.5 | *70.9* | 46.6 | 56.3 | *49.7* | 42.6 | 45.9 |
| ≥ 7 (unanimous) | **57.1** | 38.2 | 45.8 | **80.5** | 33.2 | 47.0 | **56.0** | 31.2 | 40.1 |

### Causal event retrieval by sentence scope

The runs of the causal event retrieval experiment of the paper: DeepSeek-V4-Flash with
$N=7$ resamplings, one example per prompt drawn from a pool of 20 annotated documents, the
`*_lean_coref` prompts on EventStoryLine and MAVEN-ERE and `causal_timebank_syntax` on
Causal-TimeBank. Each query is an event mention annotated as the effect of at least one
gold causal relation, and its relevant candidates are its gold causes. Its candidates are
restricted to the other mentions of its sentence (intra-sentence) or to the mentions of the
other sentences (inter-sentence). Causal-TimeBank has no inter-sentence relations. Next to
each corpus: number of queries (q.) and mean number of candidates per query (cand.).

| Ranker | MRR | H@1 | H@5 | R@1 | R@5 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **Intra-sentence: candidates in the query's sentence** | | | | | | | |
| **CTB (280 q., 3.6 cand.)** | | | | | | | |
| Vote count | **87.1** | **79.3** | **99.6** | **76.6** | **99.5** | **90.1** | **90.3** |
| TF-IDF cosine | 55.3 | 26.8 | 98.9 | 26.1 | 98.8 | 66.2 | 66.7 |
| Embedding cosine | 55.4 | 27.5 | 99.3 | 26.6 | 99.3 | 66.5 | 66.8 |
| Word distance | 55.5 | 24.6 | 97.9 | 23.8 | 97.5 | 66.1 | 67.0 |
| Random | 60.7 | 37.1 | 97.9 | 35.4 | 97.7 | 69.6 | 70.5 |
| **ESC (1,267 q., 4.1 cand.)** | | | | | | | |
| Vote count | **85.1** | **75.4** | **98.7** | **67.3** | **97.6** | **87.5** | **88.4** |
| TF-IDF cosine | 62.9 | 40.9 | 95.3 | 35.8 | 93.6 | 69.8 | 72.0 |
| Embedding cosine | 62.5 | 40.3 | 94.6 | 35.4 | 92.9 | 69.3 | 71.8 |
| Word distance | 65.2 | 42.5 | 97.9 | 37.5 | 96.3 | 72.6 | 73.9 |
| Random | 63.1 | 42.1 | 95.7 | 36.2 | 93.6 | 69.6 | 71.9 |
| **MAVEN-ERE (2,777 q., 2.7 cand.)** | | | | | | | |
| Vote count | **89.7** | **82.0** | **99.5** | **73.3** | **99.2** | **91.8** | **92.1** |
| TF-IDF cosine | 73.5 | 54.5 | 99.0 | 48.2 | 98.6 | 79.7 | 80.2 |
| Embedding cosine | 73.0 | 53.8 | 99.0 | 47.9 | 98.7 | 79.5 | 80.0 |
| Word distance | 75.4 | 56.6 | 99.4 | 51.0 | 98.9 | 81.4 | 81.8 |
| Random | 74.8 | 57.1 | 99.0 | 49.9 | 98.6 | 80.5 | 81.1 |
| **Inter-sentence: candidates in the other sentences** | | | | | | | |
| **ESC (1,304 q., 20.5 cand.)** | | | | | | | |
| Vote count | **70.5** | **58.4** | **86.5** | **30.3** | **72.3** | **66.3** | **70.6** |
| TF-IDF cosine | 43.4 | 21.4 | 72.4 | 11.0 | 45.9 | 35.8 | 44.9 |
| Embedding cosine | 43.5 | 23.0 | 71.4 | 10.5 | 46.0 | 35.5 | 45.0 |
| Word distance | 39.6 | 19.9 | 66.3 | 9.3 | 42.7 | 32.1 | 41.5 |
| Random | 34.4 | 15.3 | 56.6 | 7.0 | 34.2 | 26.0 | 36.0 |
| **MAVEN-ERE (3,990 q., 30.3 cand.)** | | | | | | | |
| Vote count | **53.2** | **40.8** | **66.9** | **21.7** | **51.5** | **46.8** | **52.5** |
| TF-IDF cosine | 32.3 | 17.7 | 46.6 | 7.9 | 31.6 | 25.3 | 32.8 |
| Embedding cosine | 29.5 | 14.8 | 44.2 | 7.1 | 29.0 | 22.6 | 30.3 |
| Word distance | 30.3 | 14.7 | 46.6 | 8.1 | 32.2 | 24.5 | 31.9 |
| Random | 26.1 | 10.5 | 41.8 | 4.9 | 24.1 | 18.2 | 26.3 |

### Causal event retrieval, undirected

The same runs with direction collapsed: a query is any event mention in a gold causal
relation, its relevant candidates are its gold partners in either direction, and a
candidate is scored by its votes in both directions. Candidates are all the other mentions
of the document.

| Ranker | MRR | H@1 | H@5 | R@1 | R@5 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **CTB (556 q., 81.4 cand.)** | | | | | | | |
| Vote count | **77.5** | **74.6** | 79.1 | **71.9** | 78.8 | **76.9** | **77.8** |
| TF-IDF cosine | 54.1 | 25.2 | 97.5 | 23.9 | 97.4 | 64.8 | 65.7 |
| Embedding cosine | 55.1 | 27.3 | **98.4** | 26.0 | **98.4** | 65.8 | 66.4 |
| Word distance | 51.4 | 21.0 | 94.8 | 19.9 | 94.2 | 61.7 | 63.7 |
| Random | 10.1 | 3.1 | 12.2 | 2.9 | 11.6 | 7.3 | 10.5 |
| **ESC (2,682 q., 24.3 cand.)** | | | | | | | |
| Vote count | **69.7** | **57.2** | 86.9 | **22.6** | **65.2** | **63.8** | **68.9** |
| TF-IDF cosine | 58.8 | 38.1 | 88.0 | 14.7 | 53.9 | 47.6 | 54.9 |
| Embedding cosine | 59.3 | 39.0 | 88.1 | 15.5 | 53.7 | 47.8 | 55.6 |
| Word distance | 60.8 | 40.1 | **89.5** | 16.4 | 54.1 | 48.7 | 55.7 |
| Random | 38.2 | 20.6 | 59.7 | 6.3 | 28.4 | 25.9 | 34.7 |
| **MAVEN-ERE (9,143 q., 33.1 cand.)** | | | | | | | |
| Vote count | **61.0** | **48.3** | **76.1** | **25.8** | **55.9** | **53.0** | **57.9** |
| TF-IDF cosine | 51.1 | 34.4 | 70.5 | 19.4 | 51.9 | 44.5 | 49.5 |
| Embedding cosine | 51.0 | 34.7 | 70.2 | 20.2 | 51.5 | 44.6 | 49.6 |
| Word distance | 51.5 | 33.9 | 73.6 | 19.8 | 54.8 | 46.2 | 51.5 |
| Random | 26.6 | 12.0 | 41.3 | 4.4 | 21.7 | 17.7 | 24.8 |

### Causal event retrieval by sentence scope, undirected

The undirected setting, with candidates restricted by sentence scope as above.

| Ranker | MRR | H@1 | H@5 | R@1 | R@5 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **Intra-sentence: candidates in the query's sentence** | | | | | | | |
| **CTB (556 q., 3.6 cand.)** | | | | | | | |
| Vote count | **87.3** | **80.0** | **98.7** | **76.8** | **98.7** | **89.9** | **90.4** |
| TF-IDF cosine | 53.9 | 24.8 | 97.5 | 23.5 | 97.4 | 64.6 | 65.6 |
| Embedding cosine | 55.1 | 27.3 | 98.4 | 26.0 | 98.4 | 65.8 | 66.4 |
| Word distance | 56.3 | 26.3 | 98.0 | 24.8 | 97.8 | 66.7 | 67.5 |
| Random | 61.7 | 37.9 | 96.9 | 35.9 | 96.7 | 70.0 | 71.2 |
| **ESC (2,260 q., 4.0 cand.)** | | | | | | | |
| Vote count | **84.0** | **73.3** | 98.6 | **60.3** | **97.3** | **86.4** | **87.4** |
| TF-IDF cosine | 65.6 | 44.2 | 96.5 | 35.6 | 94.3 | 72.1 | 74.1 |
| Embedding cosine | 66.4 | 45.7 | 96.6 | 37.1 | 94.4 | 72.6 | 74.8 |
| Word distance | 69.5 | 48.4 | **99.0** | 39.0 | 97.1 | 75.7 | 76.9 |
| Random | 66.3 | 46.1 | 96.8 | 36.6 | 94.0 | 72.0 | 74.3 |
| **MAVEN-ERE (5,108 q., 2.7 cand.)** | | | | | | | |
| Vote count | **88.3** | **79.4** | **99.6** | **67.2** | **99.3** | **90.8** | **91.0** |
| TF-IDF cosine | 75.0 | 57.0 | 99.2 | 47.6 | 98.7 | 81.0 | 81.5 |
| Embedding cosine | 75.6 | 57.9 | 99.4 | 48.8 | 98.8 | 81.4 | 81.9 |
| Word distance | 78.6 | 61.9 | **99.6** | 52.6 | 99.2 | 83.9 | 84.2 |
| Random | 78.0 | 62.0 | 99.2 | 52.0 | 98.7 | 83.0 | 83.5 |
| **Inter-sentence: candidates in the other sentences** | | | | | | | |
| **ESC (2,006 q., 20.0 cand.)** | | | | | | | |
| Vote count | **71.7** | **58.9** | **88.8** | **25.9** | **69.4** | **66.3** | **71.9** |
| TF-IDF cosine | 50.3 | 28.3 | 79.1 | 11.7 | 46.5 | 40.2 | 49.2 |
| Embedding cosine | 50.5 | 29.0 | 78.8 | 11.3 | 47.0 | 40.4 | 49.8 |
| Word distance | 47.4 | 27.4 | 75.4 | 10.7 | 44.1 | 37.2 | 46.8 |
| Random | 41.6 | 22.9 | 65.8 | 8.6 | 36.3 | 31.0 | 41.0 |
| **MAVEN-ERE (6,167 q., 29.7 cand.)** | | | | | | | |
| Vote count | **53.5** | **39.0** | **70.8** | **17.0** | **49.0** | **45.9** | **51.9** |
| TF-IDF cosine | 32.1 | 17.0 | 47.4 | 6.7 | 28.8 | 24.2 | 32.3 |
| Embedding cosine | 31.4 | 15.9 | 46.9 | 6.5 | 27.8 | 23.4 | 31.6 |
| Word distance | 36.8 | 20.8 | 54.9 | 9.0 | 35.8 | 29.9 | 37.4 |
| Random | 30.4 | 14.4 | 47.7 | 5.1 | 25.0 | 21.0 | 29.1 |
