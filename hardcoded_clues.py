"""Domain- and number-aware lookup over HARDCODED_CLUES (see hardcoded_clues_data.py)."""

import random

import numpy as np
from sentence_transformers import SentenceTransformer

from hardcoded_clues_data import HARDCODED_CLUES

_SIMILARITY_THRESHOLD = 0.45
_DIFF_RATIO_THRESHOLD = 0.15

_embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

_ALL_TAGS = sorted({tag for entries in HARDCODED_CLUES.values() for _, tags in entries for tag in tags})
_tag_embeddings = _embed_model.encode(_ALL_TAGS, normalize_embeddings=True)
_tag_index = {tag: i for i, tag in enumerate(_ALL_TAGS)}


def get_hardcoded_clue(number, domain):
    """The file hard_coded_clues_data.py contains a dictionary of hard-coded clues
    indexed by target number and domain.  
    The goal is to find a hard-coded clue whose domain (kinda) matches the input
    domain and number.  

    1. We prioritize domain matching over number matching. To do this, we use
       cosine similarity between the input domain and the domains of the hard-coded clues (tags). 
       Select the tag that best matches the input domain. 
    2. Look up all hard-coded clues with that tag. 
       Keep those whose target number is within a threshold from the input number:
       diff = input number - hard-coded clue's target number 
       We want those with diff/(input number) < _DIFF_RATIO_THRESHOLD
    3. If there are many clue candidates, pick one randomly and return 
       (clue, diff). Else, return (None, None)
    """
    domain_vec = _embed_model.encode(domain.lower().strip(), normalize_embeddings=True)
    sims = _tag_embeddings @ domain_vec

    scored = []  # (score, diff, clue_text)
    for key, entries in HARDCODED_CLUES.items():
        diff = number - key
        for clue_text, tags in entries:
            score = max(sims[_tag_index[tag]] for tag in tags)
            if score < _SIMILARITY_THRESHOLD:
                continue
            scored.append((score, diff, clue_text))

    if not scored:
        return None, None

    best_score = max(score for score, _, _ in scored)
    candidates = [
        (diff, clue_text)
        for score, diff, clue_text in scored
        if score == best_score and abs(diff) / max(abs(number), 1) <= _DIFF_RATIO_THRESHOLD
    ]
    if not candidates:
        return None, None

    diff, clue_text = random.choice(candidates)
    return clue_text, diff
