"""P9 - lexical + structural retrieval over the local corpus. No embeddings, no network.

BM25 supplies the "does this document talk about what I asked" signal; additive structural
boosts supply the "is this document about *my* factor" signal — shared operators, the same
universe, the same enabled categories, recency. Embeddings were deliberately not used: the
corpus is small, entirely local, and the retrieval has to be deterministic and unit-testable so
an explanation can be reproduced from a stored prompt fingerprint.

Selection is greedy under a character budget, with per-kind quotas so one chatty document kind
(there are hundreds of primitives) cannot crowd out the handful of session documents that carry
the actual history.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from alphalineage.explain.corpus import Document, tokenize

_K1 = 1.5
_B = 0.75

#: Additive boosts applied after BM25. Deliberately modest: lexical relevance leads, structure
#: breaks ties.
_OPERATOR_OVERLAP_WEIGHT = 2.5
_UNIVERSE_MATCH = 1.5
_CATEGORY_OVERLAP_WEIGHT = 0.75
_RECENCY_BONUS = 0.5

#: Multiplicative prior per document kind. Reference material (primitives, indicators) is already
#: summarized directly into the prompt's operator-registry section, so retrieving more of it is
#: partly redundant; workspace history is not available anywhere else in the prompt.
KIND_PRIORS: dict[str, float] = {
    "genetics": 1.6,
    "round": 1.5,
    "session": 1.4,
    "factor": 1.2,
    "indicator": 1.0,
    "primitive": 0.8,
}

#: Default share of the context budget each kind may occupy. History outranks reference material
#: because reference material is also summarized directly into the prompt's registry section.
DEFAULT_QUOTAS: dict[str, float] = {
    "session": 0.20,
    "round": 0.20,
    "genetics": 0.25,
    "factor": 0.15,
    "indicator": 0.10,
    "primitive": 0.10,
}


@dataclass(frozen=True)
class Hit:
    """One retrieved document with its score decomposition (shown in the prompt preview)."""

    document: Document
    score: float
    lexical: float
    structural: float
    matched_terms: tuple[str, ...]
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.document.id,
            "kind": self.document.kind,
            "title": self.document.title,
            "score": round(self.score, 4),
            "lexical": round(self.lexical, 4),
            "structural": round(self.structural, 4),
            "matched_terms": list(self.matched_terms),
            "characters": len(self.document.text),
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class Query:
    """What we are retrieving for: free text plus the target factor's structural fingerprint.

    ``pinned`` names documents that are *definitionally* about the target — the session it came
    from, its own segment, that segment's evolution trajectory. Those must not have to win a
    similarity contest against reference material that happens to share operator vocabulary, so
    they bypass ranking and are placed first.
    """

    text: str = ""
    operators: frozenset[str] = frozenset()
    universe: str = ""
    categories: frozenset[str] = frozenset()
    pinned: frozenset[str] = frozenset()

    def terms(self) -> list[str]:
        parts = [self.text, " ".join(sorted(self.operators))]
        if self.universe:
            parts.append(self.universe)
        return tokenize(" ".join(parts))


class Index:
    """A BM25 index over a document list. Built per request; the corpus is small."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        self._term_frequencies: list[Counter[str]] = [Counter(doc.tokens) for doc in self.documents]
        self._lengths = [max(len(doc.tokens), 1) for doc in self.documents]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        self._document_frequency = document_frequency

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._document_frequency.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, terms: Sequence[str], position: int) -> tuple[float, tuple[str, ...]]:
        frequencies = self._term_frequencies[position]
        length = self._lengths[position]
        total = 0.0
        matched: list[str] = []
        for term in dict.fromkeys(terms):
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            idf = self._idf(term)
            denominator = tf + _K1 * (1 - _B + _B * length / self._avg_length)
            total += idf * (tf * (_K1 + 1)) / denominator
            matched.append(term)
        return total, tuple(matched)


def _structural_score(doc: Document, query: Query, newest: str) -> float:
    score = 0.0
    if query.operators and doc.operators:
        overlap = len(query.operators & doc.operators)
        if overlap:
            score += _OPERATOR_OVERLAP_WEIGHT * (overlap / len(query.operators)) ** 0.5
    if query.universe and doc.universe and query.universe == doc.universe:
        score += _UNIVERSE_MATCH
    if query.categories and doc.categories:
        overlap = len(query.categories & doc.categories)
        if overlap:
            score += _CATEGORY_OVERLAP_WEIGHT * overlap / len(query.categories)
    if newest and doc.created_at and doc.created_at >= newest:
        score += _RECENCY_BONUS
    return score


def rank(documents: Sequence[Document], query: Query) -> list[Hit]:
    """Score every document; ties break on document id so results are reproducible.

    Pinned documents are always returned, ahead of everything else, even if they match no query
    term at all.
    """
    if not documents:
        return []
    index = Index(documents)
    terms = query.terms()
    newest = max((doc.created_at for doc in documents if doc.created_at), default="")
    pinned_hits: list[Hit] = []
    hits: list[Hit] = []
    for position, doc in enumerate(documents):
        lexical, matched = index.score(terms, position)
        structural = _structural_score(doc, query, newest)
        total = (lexical + structural) * KIND_PRIORS.get(doc.kind, 1.0)
        is_pinned = doc.id in query.pinned
        if total <= 0 and not is_pinned:
            continue
        hit = Hit(
            document=doc,
            score=total,
            lexical=lexical,
            structural=structural,
            matched_terms=matched[:8],
            pinned=is_pinned,
        )
        (pinned_hits if is_pinned else hits).append(hit)
    pinned_hits.sort(key=lambda hit: (-hit.score, hit.document.id))
    hits.sort(key=lambda hit: (-hit.score, hit.document.id))
    return pinned_hits + hits


def select(
    hits: Iterable[Hit],
    *,
    budget_chars: int = 12_000,
    quotas: dict[str, float] | None = None,
    max_documents: int = 24,
) -> list[Hit]:
    """Greedy budgeted selection with per-kind quotas.

    A kind may exceed its quota only once every other kind has had its chance, so a workspace
    with no sessions yet still fills the budget with useful reference material.
    """
    shares = quotas or DEFAULT_QUOTAS
    remaining = {kind: int(budget_chars * share) for kind, share in shares.items()}
    ordered = list(hits)
    chosen: list[Hit] = []
    deferred: list[Hit] = []
    used = 0

    # Pinned documents take their budget first and ignore quotas: they are the context that is
    # actually about the target, not merely similar to it.
    for hit in ordered:
        if not hit.pinned:
            continue
        size = len(hit.document.text)
        if used + size <= budget_chars:
            remaining[hit.document.kind] = max(0, remaining.get(hit.document.kind, 0) - size)
            chosen.append(hit)
            used += size

    for hit in ordered:
        if hit.pinned:
            continue
        if len(chosen) >= max_documents:
            break
        size = len(hit.document.text)
        allowance = remaining.get(hit.document.kind, 0)
        if size <= allowance and used + size <= budget_chars:
            remaining[hit.document.kind] = allowance - size
            chosen.append(hit)
            used += size
        else:
            deferred.append(hit)

    for hit in deferred:
        if len(chosen) >= max_documents:
            break
        size = len(hit.document.text)
        if used + size <= budget_chars:
            chosen.append(hit)
            used += size

    pinned = [hit for hit in chosen if hit.pinned]
    rest = sorted(
        (hit for hit in chosen if not hit.pinned),
        key=lambda hit: (-hit.score, hit.document.id),
    )
    return pinned + rest


def retrieve(
    documents: Sequence[Document],
    query: Query,
    *,
    budget_chars: int = 12_000,
    max_documents: int = 24,
    quotas: dict[str, float] | None = None,
) -> list[Hit]:
    """Rank then budget. The single entry point used by the prompt builder."""
    return select(
        rank(documents, query),
        budget_chars=budget_chars,
        quotas=quotas,
        max_documents=max_documents,
    )
