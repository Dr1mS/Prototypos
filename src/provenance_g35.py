"""provenance_g35.py -- G3.5 subagent A: retrieval-provenance analysis helpers.

Pure Python: ZERO LLM calls, ZERO daemon calls, no numpy. Operates on the
records captured by memory_g3.RETRIEVAL_SINK (the pure-logging seam added for
G3.5; contract_g35.py "RETRIEVAL PROVENANCE"), each of shape

    {"store_size": <len(entries) at query time>,
     "query":      <query[:80]>,          # 80 == contract_g35.SINK_QUERY_PREFIX
     "top":        <selected store indices, oldest-first>,
     "cos":        <cosine per selected index, same order>,
     "k":          <requested k>}

ORIGIN-TURN MAPPING (load-bearing; frozen in contract_g35.py):

    store index i  ==  origin turn i + 1

On the G3/G3.5 rungs exactly ONE entry is appended per life turn, every life
starts from an EMPTY store, entries are append-only, and probing is read-only
(battery25 snapshot/restore, proven bit-identical) -- so the 0-based append
index i of a stored entry identifies exactly the 1-based life turn i+1 that
wrote it. Sink records carry store INDICES (0-based); frac_origin and
origin_counts speak in ORIGIN TURNS (1-based). E.g. on the hyst arm
(contract_g35: 15 permissive + 15 caution + 5 neutral), origin turns 1..15 are
the soft (induction-phase) exemplars and 16..35 the firm/tail exemplars.

PROBE-vs-TURN CLASSIFICATION (frozen, contract_g35.py): a record is a PROBE
retrieval iff its (already [:80]-truncated) query equals probe["text"][:80]
for one of contract_g2.PROBES -- exact string match. Structural basis: the
battery probes call respond() with the VERBATIM probe text, and life turns
call it with the VERBATIM user turn text, so the sink query prefix separates
the two exactly (no turn text collides with a probe prefix).

Frozen quantities these helpers feed (contract_g35.py):

    firm_frac_ret(t) = frac_origin(battery_records(records, t), 16, 35)
    soft_frac_ret(t) = frac_origin(battery_records(records, t),  1, 15)

Ownership: subagent A. Verified daemon-free by selftest_a35.py.
"""
from __future__ import annotations

from contract_g2 import PROBES
from contract_g35 import SINK_QUERY_PREFIX

# The exact 80-char probe prefixes the sink stores when a battery probe
# retrieves (probes query with the verbatim probe text; the sink truncates).
_PROBE_ID_BY_PREFIX = {p["text"][:SINK_QUERY_PREFIX]: p["id"] for p in PROBES}


def classify(records):
    """Label each sink record as a probe or a turn retrieval.

    Returns a NEW list of dicts: each is a copy of the input record (the
    "top"/"cos" lists are copied too -- the output never aliases the input's
    lists) plus two added keys:

        "kind":     "probe" if the record's query exactly equals
                    probe["text"][:80] for one of contract_g2.PROBES,
                    else "turn";
        "probe_id": the matching probe's id (e.g. "deploy-prod"), or None
                    for turn retrievals.

    The input records are never mutated. Exact string match only -- a near-miss
    (e.g. a 79-char prefix) is a turn, by design: probes query with the
    VERBATIM probe text, so equality on the [:80] prefix is exact.
    """
    out = []
    for rec in records:
        new = dict(rec)
        if isinstance(new.get("top"), list):
            new["top"] = list(new["top"])
        if isinstance(new.get("cos"), list):
            new["cos"] = list(new["cos"])
        probe_id = _PROBE_ID_BY_PREFIX.get(rec["query"])
        new["kind"] = "probe" if probe_id is not None else "turn"
        new["probe_id"] = probe_id
        out.append(new)
    return out


def battery_records(records, t):
    """The probe-kind records captured at store size t.

    `t` is the battery's probe point in turns: because one entry is appended
    per turn and probing is read-only, the battery run after turn t sees
    len(entries) == t, so its probe retrievals log store_size == t.

    Accepts raw sink records (classified here first) OR already-classified
    records (any record missing "kind" triggers a full classify pass).
    Returns a list of classified records; [] when none match.
    """
    if not all("kind" in r for r in records):
        records = classify(records)
    return [r for r in records
            if r["kind"] == "probe" and r["store_size"] == t]


def frac_origin(records_subset, lo, hi):
    """Fraction of retrieved items whose ORIGIN TURN lies in [lo, hi].

    Pools every index of every record's "top" across `records_subset`, maps
    each store index i to its origin turn i+1 (see module docstring), and
    returns the fraction with lo <= i+1 <= hi (both ends inclusive).

    Returns None when the pooled set is empty (no records, or only empty
    "top" lists) -- the caller must not silently treat that as 0.0.
    """
    total = 0
    hits = 0
    for rec in records_subset:
        for i in rec["top"]:
            total += 1
            if lo <= i + 1 <= hi:
                hits += 1
    if total == 0:
        return None
    return hits / total


def origin_counts(records_subset):
    """Histogram of retrieved items by ORIGIN TURN (for the provenance figure).

    Pools every index of every record's "top" across `records_subset` and
    returns {origin_turn: count} with origin_turn == store index + 1 (1-based;
    see module docstring). Turns never retrieved are absent from the dict.
    Returns {} on empty input.
    """
    counts = {}
    for rec in records_subset:
        for i in rec["top"]:
            turn = i + 1
            counts[turn] = counts.get(turn, 0) + 1
    return counts


__all__ = [
    "classify",
    "battery_records",
    "frac_origin",
    "origin_counts",
]
