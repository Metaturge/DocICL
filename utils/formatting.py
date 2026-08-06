from __future__ import annotations

import json
from typing import List, Tuple

from utils.labels import NOREL_VARIANTS

# Injected into the prompt in place of an explicit candidate list: the model enumerates
# the pairs itself and omission encodes norel.
PAIR_LINES = "Predict all event mention pairs in the text; omit a pair to mark it norel."


def format_gold_output(gold_triples: List[Tuple[str, str, str]]) -> str:
    """
    Formats gold triples as the JSON output the model is expected to produce.

    The task is open-ended: the model only outputs positive pairs and omission encodes
    norel, so norel triples are filtered out of gold.
    """
    items = [
        {"pair": f"{src},{tgt}", "label": lbl}
        for src, lbl, tgt in gold_triples
        if lbl.lower() not in NOREL_VARIANTS
    ]
    return json.dumps(items)
