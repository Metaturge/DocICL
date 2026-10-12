# rankers.py
"""Causal event retrieval from DocICL vote counts, against non-LLM rankers.

Each event mention annotated as the effect of at least one gold causal relation is a query.
Its candidates are the other event mentions of its document, and a candidate c is relevant
if the gold annotation holds c -> q. Rankers:

  vote count                  number of resamplings that predict c -> q, ties ranked randomly
  vote count + word distance  the same, ties ranked by word distance instead
  TF-IDF / embedding cosine   similarity between the "trigger: sentence" strings of q and c
  word distance               number of tokens between the first tokens of q and c, closest first
  random                      chance level

Every ranker first shuffles the candidates with the same seeded generator, so any tie it
leaves is ranked randomly, identically for all rankers.

Settings: scope "all" (every other mention), "intra" (the query's sentence only) or "inter"
(the other sentences only), each directed (relevant = gold causes of q) or undirected
(relevant = gold causes and effects of q, votes summed over both directions).
Metrics: MRR, Hit@k, Recall@k and nDCG@k (binary gains), k = 1..ranking.k_max.

Votes come from the run report: a pass asserts c -> q by "c,q causes" or by "q,c causedby",
so the votes of c -> q are the larger of the two tallies.

Usage:
  uv run rankers.py <run> [<run> ...]

<run> is a report path or any fragment of its name in logs/allatonce/ (e.g. its 8-hex id).
Results go to <report stem>.ranking.json next to the report. Settings: `ranking` in config.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import yaml
from tabulate import tabulate

from dataprep.dataprep import REPO_ROOT, parse_annotations
from utils.labels import CAUSED_BY, CAUSES

LOG_DIR = REPO_ROOT / "logs" / "allatonce"
SCOPES = ("all", "intra", "inter")
# Columns of the printed table: (header, metric, k); the JSON holds every k.
COLUMNS = (("MRR", "mrr", None), ("H@1", "hit", 1), ("H@5", "hit", 5), ("R@1", "recall", 1),
           ("R@5", "recall", 5), ("nDCG@5", "ndcg", 5), ("nDCG@10", "ndcg", 10))


@dataclass
class Doc:
    id: str
    mentions: List[str]
    sentence_of: Dict[str, int]
    first_token: Dict[str, int]
    texts: Dict[str, str]           # mention -> "trigger: sentence"
    gold: Set[Tuple[str, str]]      # (cause, effect)


# =============================================================================
# LOADING
# =============================================================================

def find_report(selector: str) -> Path:
    path = Path(selector)
    if path.is_file():
        return path
    hits = [p for p in LOG_DIR.glob(f"run_*{selector}*.json") if not p.name.endswith(".ranking.json")]
    if len(hits) != 1:
        raise SystemExit(f"'{selector}' matches {len(hits)} reports in {LOG_DIR}: {[h.name for h in hits]}")
    return hits[0]


def load_docs(ds_cfg: dict, doc_ids: List[str]) -> List[Doc]:
    """The run's documents, in doc_ids order, from the dataset split the run evaluated."""
    data_dir = Path(ds_cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    rows = json.loads((data_dir / f"{ds_cfg['split']}.json").read_text(encoding="utf-8"))
    by_id = {str(row["id"]): row for row in rows}
    docs = []
    for doc_id in doc_ids:
        row = by_id[doc_id]
        tokens, sentences = row["tokens"], row["sentences"]
        sentence_of, first_token, texts = {}, {}, {}
        for m, span in zip(row["mentions"], row["spans"]):
            first_token[m] = span[0]
            sentence_of[m] = next(i for i, (start, end) in enumerate(sentences) if start <= span[0] < end)
            start, end = sentences[sentence_of[m]]
            texts[m] = f"{' '.join(tokens[t] for t in span)}: {' '.join(tokens[start:end])}"
        gold = {(a, b) if lbl == CAUSES else (b, a)
                for a, lbl, b in parse_annotations(row[ds_cfg["ann_field"]]) if lbl in (CAUSES, CAUSED_BY)}
        docs.append(Doc(doc_id, list(row["mentions"]), sentence_of, first_token, texts, gold))
    return docs


def load_run(path: Path):
    """(config, n_runs, documents, {doc_id: Counter({(cause, effect): votes})})."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg, res = payload["config"], payload["results"]
    resampling = cfg["experiment"]["resampling"]
    n_runs = resampling["n_runs"] if resampling["enabled"] else 1
    votes: Dict[str, Counter] = defaultdict(Counter)
    for row in res["per_pair_predictions"]:
        a, b = (x.strip() for x in row["pair"].split(",", 1))
        tally = row["vote_counts"] or {}
        v = votes[row["id"]]
        v[(a, b)] = max(v[(a, b)], tally.get(CAUSES, 0))
        v[(b, a)] = max(v[(b, a)], tally.get(CAUSED_BY, 0))
    doc_ids = sorted({m["id"] for m in res["per_doc_metrics"]})
    docs = load_docs(cfg["datasets"][cfg["active_dataset"]], doc_ids)
    return cfg, n_runs, docs, votes


# =============================================================================
# RANKERS
# =============================================================================

def cosine_scorers(docs: List[Doc], embedding_model: str, device: str) -> Dict[str, Callable]:
    """TF-IDF (fit on every mention text of the run) and sentence-embedding cosine."""
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [t for d in docs for t in d.texts.values()]
    model = SentenceTransformer(embedding_model, device=device)
    encoders = {
        "TF-IDF cosine": lambda: TfidfVectorizer().fit_transform(texts),
        "Embedding cosine": lambda: model.encode(texts, convert_to_numpy=True,
                                                 normalize_embeddings=True, batch_size=64),
    }
    scorers = {}
    for name, encode in encoders.items():
        X, sims, row = encode(), {}, 0
        for d in docs:
            block = X[row:row + len(d.texts)]
            S = block @ block.T
            sims[d.id] = ({m: i for i, m in enumerate(d.texts)}, S.toarray() if hasattr(S, "toarray") else S)
            row += len(d.texts)
        scorers[name] = lambda d, q, c, sims=sims: float(sims[d.id][1][sims[d.id][0][q], sims[d.id][0][c]])
    return scorers


def word_distance(d: Doc, q: str, c: str) -> int:
    return -abs(d.first_token[c] - d.first_token[q])


def queries(d: Doc, scope: str, directed: bool):
    """[(query, candidates, relevant)] of one document."""
    def keep(a, b):
        return scope == "all" or (d.sentence_of[a] == d.sentence_of[b]) == (scope == "intra")

    relevant = defaultdict(set)
    for c, e in d.gold:
        if keep(c, e):
            relevant[e].add(c)
            if not directed:
                relevant[c].add(e)
    out = []
    for q in d.mentions:
        cands = [m for m in d.mentions if m != q and keep(m, q)]
        rel = relevant[q] & set(cands)
        if rel:
            out.append((q, cands, rel))
    return out


# =============================================================================
# METRICS
# =============================================================================

def _dcg(ranks) -> float:
    return sum(1.0 / math.log2(r + 1) for r in ranks)


def evaluate(qs, score: Callable, k_max: int, seed: int) -> dict:
    """MRR, Hit@k, Recall@k and nDCG@k (in %) of one ranker over [(doc, queries)]."""
    rng = random.Random(seed)
    out = {"mrr": 0.0, "hit": [0.0] * k_max, "recall": [0.0] * k_max, "ndcg": [0.0] * k_max}
    n = 0
    for d, doc_queries in qs:
        for q, cands, relevant in doc_queries:
            order = cands[:]
            rng.shuffle(order)
            order.sort(key=lambda c: score(d, q, c), reverse=True)   # stable: ties keep the shuffle
            ranks = sorted(i + 1 for i, c in enumerate(order) if c in relevant)
            out["mrr"] += 1.0 / ranks[0]
            for k in range(1, k_max + 1):
                found = [r for r in ranks if r <= k]
                out["hit"][k - 1] += bool(found)
                out["recall"][k - 1] += len(found) / len(ranks)
                out["ndcg"][k - 1] += _dcg(found) / _dcg(range(1, min(k, len(ranks)) + 1))
            n += 1
    return {key: [100 * x / n for x in v] if isinstance(v, list) else 100 * v / n for key, v in out.items()}


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="report paths or fragments of report names in logs/allatonce/")
    args = ap.parse_args()
    rcfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))["ranking"]
    k_max, seed = rcfg["k_max"], rcfg["seed"]

    baselines = {}   # (dataset, doc ids) -> cosine scorers, shared by runs on the same documents
    for selector in args.runs:
        path = find_report(selector)
        cfg, n_runs, docs, votes = load_run(path)
        dataset = cfg["active_dataset"]
        key = (dataset, tuple(d.id for d in docs))
        if key not in baselines:
            baselines[key] = cosine_scorers(docs, rcfg["embedding_model"], rcfg["embedding_device"])

        result = {"report": path.name, "dataset": dataset, "n_runs": n_runs, "n_docs": len(docs), "settings": {}}
        for scope in SCOPES:
            for directed in (True, False):
                if directed:
                    vote = lambda d, q, c: votes[d.id][(c, q)]
                else:
                    vote = lambda d, q, c: votes[d.id][(c, q)] + votes[d.id][(q, c)]
                rankers = {
                    f"Vote count (N={n_runs})": vote,
                    f"Vote count (N={n_runs}) + word distance": lambda d, q, c, vote=vote: (vote(d, q, c), word_distance(d, q, c)),
                    **baselines[key],
                    "Word distance": word_distance,
                    "Random": lambda d, q, c: 0,
                }
                qs = [(d, queries(d, scope, directed)) for d in docs]
                flat = [(d, q, rel) for d, dq in qs for q, _, rel in dq]
                setting = f"{scope}/{'directed' if directed else 'undirected'}"
                if not flat:
                    print(f"\n{dataset} {path.stem.rsplit('_', 1)[-1]} (N={n_runs}), {setting}: no queries")
                    continue
                result["settings"][setting] = {
                    "n_queries": len(flat),
                    "mean_candidates": sum(len(c) for _, dq in qs for _, c, _ in dq) / len(flat),
                    "mean_relevant": sum(len(rel) for _, _, rel in flat) / len(flat),
                    "pct_voted_relevant": 100 * sum(any(vote(d, q, c) > 0 for c in rel) for d, q, rel in flat) / len(flat),
                    "rankers": {name: evaluate(qs, s, k_max, seed) for name, s in rankers.items()},
                }
                st = result["settings"][setting]
                print(f"\n{dataset} {path.stem.rsplit('_', 1)[-1]} (N={n_runs}), {setting}: {st['n_queries']} queries, "
                      f"{st['mean_candidates']:.1f} candidates and {st['mean_relevant']:.2f} relevant per query, "
                      f"{st['pct_voted_relevant']:.1f}% with a voted relevant candidate")
                print(tabulate([[name] + [m[metric] if k is None else m[metric][k - 1] for _, metric, k in COLUMNS]
                                for name, m in st["rankers"].items()],
                               headers=["Ranker"] + [h for h, _, _ in COLUMNS], floatfmt=".1f"))

        out = path.with_name(path.stem + ".ranking.json")
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
