
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Set, List, Dict, Any, Optional, Iterator, Tuple

# Repo root — data_dir values in config.yaml are relative to it.
REPO_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# REGEX CONSTANTS
# =============================================================================

# Matches a relation triple in the annotation string.
# Pattern: <ID CONTENT> LABEL <ID CONTENT>
# Constraints:
#   1. ID cannot contain whitespace or '>'.
#   2. CONTENT cannot contain newlines.
# We capture:
#   Group 1: Source ID
#   Group 3: Relation Label
#   Group 4: Target ID
# Groups 2 and 5 (Content) are matched to ensure validity but ignored in output.
RELATION_TRIPLE_RE = re.compile(
    r"<([^>\s]+)\s+([^>\n]+)>"       # Source Tag: <ID Content>
    r"\s+"                           # Space
    r"([A-Za-z0-9_\-]+)"             # Relation Label
    r"\s+"                           # Space
    r"<([^>\s]+)\s+([^>\n]+)>"       # Target Tag: <ID Content>
)


def parse_annotations(
    ann_text: str, 
    valid_labels: Optional[Set[str]] = None
) -> List[Tuple[str, str, str]]:
    """
    Parses an annotation string containing relations into triples.
    Labels and IDs are extracted exactly as they appear (no normalization).
    
    Args:
        ann_text: The raw annotation string.
        valid_labels: If provided, only keep relations strictly matching these strings.

    Returns:
        List[Tuple[str, str, str]]: A list of (source_id, label, target_id) triples.
    """
    triples: List[Tuple[str, str, str]] = []
    if not ann_text:
        return triples

    for m in RELATION_TRIPLE_RE.finditer(ann_text):
        src_id = m.group(1)
        label = m.group(3)
        tgt_id = m.group(4)
        
        if valid_labels and label not in valid_labels:
            continue
            
        triples.append((src_id, label, tgt_id))

    return triples


def load_dataset_parsed(
    data_dir: str,
    split: str = "test",
    text_field: str = "text",
    ann_field: str = "annots",
    valid_labels: Optional[Set[str]] = None,
    max_examples: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Loads a local dataset split and yields the raw text and parsed relation triples.

    Reads <data_dir>/<split>.json — a JSON array of rows, one per document, each holding
    text, annots, tokens, spans, mentions and sentences. Paths are resolved relative to
    the repo root, so the same config works whatever the caller's working directory is.

    Args:
        data_dir: Directory holding the split files (e.g. "data/maven_ere").
        split: Split name, i.e. the file stem to read.
        text_field: Column name containing the document text.
        ann_field: Column name containing the relation strings.
        valid_labels: Set of exact label strings to keep.
        max_examples: Max number of examples to yield (0 = all).

    Yields:
        Dict containing:
        - "id": Row ID
        - "doc_text": Raw document text (untouched)
        - "gold_triples": List of (src_id, label, tgt_id)
    """
    path = Path(data_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}. Splits live under data/<dataset>/<split>.json; "
            f"check datasets.<key>.data_dir and split in config.yaml."
        )

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    count = 0
    for i, row in enumerate(rows):
        if max_examples > 0 and count >= max_examples:
            break
            
        doc_text = row.get(text_field, "")
        ann_text = row.get(ann_field, "")
        
        # Use existing 'id' column or fallback to index
        row_id = str(row.get("id", f"idx_{i}"))
        tokens = row.get("tokens", [])
        spans = row.get("spans", [])
        mentions = row.get("mentions", [])
        sentences = row["sentences"]

        gold_triples = parse_annotations(ann_text, valid_labels=None)
        if valid_labels:
            gold_triples = [(s, l, t) for s, l, t in gold_triples if l in valid_labels]
        
        mentions_map = {}
        mention_sentence: Dict[str, int] = {}
        if tokens and spans and mentions and len(spans) == len(mentions):
            for j, mention_id in enumerate(mentions):
                token_indices = spans[j]
                # Ensure indices are valid and gather the text parts
                text_parts = []
                for idx in token_indices:
                    if 0 <= idx < len(tokens):
                        text_parts.append(tokens[idx])

                if text_parts:
                    mentions_map[mention_id] = " ".join(text_parts)

                if token_indices:
                    first_tok = token_indices[0]
                    for sent_idx, (start, end) in enumerate(sentences):
                        if start <= first_tok < end:
                            mention_sentence[mention_id] = sent_idx
                            break

        pair_list_raw = row.get("pair_list")
        pair_list_ids: List[Tuple[str, str]] = []
        if mentions:
            if pair_list_raw is not None:
                # Dataset provides an explicit candidate pair list (e.g. Causal-TimeBank).
                for pair in pair_list_raw:
                    if len(pair) == 2:
                        a, b = int(pair[0]), int(pair[1])
                        if a < len(mentions) and b < len(mentions):
                            pair_list_ids.append((mentions[a], mentions[b]))
            else:
                # No pair_list column (e.g. EventStoryLine, MAVEN-ERE): enumerate all
                # ordered mention pairs (i, j), i ≠ j — the standard SOTA setup.
                pair_list_ids = [
                    (mentions[a], mentions[b])
                    for a in range(len(mentions))
                    for b in range(len(mentions))
                    if a != b
                ]
        yield {
            "id": row_id,
            "doc_idx": i,
            "doc_text": doc_text,
            "gold_triples": gold_triples,
            "mentions_map": mentions_map,
            "mention_sentence": mention_sentence,
            "pair_list_ids": pair_list_ids,
        }
        
        count += 1