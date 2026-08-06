# utils/resampling.py
from __future__ import annotations
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Any, Dict

from utils.labels import NOREL, NOREL_VARIANTS as _NOREL_NORM

def aggregate_run_triples(
    runs_outputs: List[List[Tuple[str, str, str]]],
    tie_breaking: str = "norel"
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    """
    Aggregates results from multiple sampling runs.
    
    Returns:
        1. Final list of (src, label, tgt) triples for the winning predictions.
        2. A dictionary containing detailed stats per pair (for logging).
    """
    pair_votes = defaultdict(list)
    num_runs = len(runs_outputs)

    for run in runs_outputs:
        # One vote per directed pair per run. A single pass can list the same pair
        # more than once (observed: a model looping and re-emitting a block of its
        # JSON output on long documents), and counting each occurrence would let one
        # pass cast several votes — inflating a pair's causal tally and pushing its
        # total past num_runs. Collapse intra-run repeats to a single vote; last-wins
        # on the label if a pass gives one pair conflicting labels.
        run_labels = {}
        for src, lab, tgt in run:
            run_labels[(src, tgt)] = lab
        for (src, tgt), lab in run_labels.items():
            pair_votes[(src, tgt)].append(lab)

    final_triples = []
    per_pair_stats = {}

    for (src, tgt) in pair_votes:
        votes = pair_votes[(src, tgt)]
        
        # Fill implicit NOREL for runs that didn't output this pair
        while len(votes) < num_runs:
            votes.append(NOREL)

        counts = Counter(votes)
        max_votes = max(counts.values())
        winners = [l for l, c in counts.items() if c == max_votes]

        if len(winners) == 1:
            final_label = winners[0]
        else:
            final_label = NOREL if tie_breaking == NOREL else random.choice(winners)
            
        if final_label.lower() not in _NOREL_NORM:
            final_triples.append((src, final_label, tgt))
            
        # Store stats for this pair (converted to dict for JSON serialization)
        key = f"{src},{tgt}"
        per_pair_stats[key] = {
            "votes": votes,
            "vote_counts": dict(counts),
            "final_label": final_label
        }

    return final_triples, per_pair_stats
