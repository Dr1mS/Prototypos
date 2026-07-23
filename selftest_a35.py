"""selftest_a35.py -- G3.5 subagent A: DRY self-tests (ZERO daemon calls).

Run:  python selftest_a35.py

Proves, with an injected deterministic embed (memory_g3's `_embed` seam) and
no ollama import, no network, no daemon:

  (a) RETRIEVAL_SINK equivalence: the identical build (10 appends) + retrieve
      sequence (8 mixed queries, two of them verbatim contract_g2.PROBES
      texts) run with the sink OFF (None) and ON ([]) yields bit-identical
      retrieve returns, bit-identical store snapshots and an identical embed
      count -- the sink is pure logging. The 8 captured records have exactly
      the frozen shape (store_size at query time, query[:80], top
      oldest-first unique ints, cos parallel to top and bit-identical to an
      independent recomputation, k as passed).
  (b) sink isolation: store_size is the query-time size; restore/reset leave
      the sink untouched; mutating a sink record's lists never touches the
      store or future retrievals (records are copies, never aliases); the
      empty-store short-circuit logs nothing; setting the sink back to None
      restores the silent default path.
  (c) provenance_g35.classify separates the two probe retrievals (correct
      probe_id, exact [:80] match; a 79-char near-miss stays a turn) from the
      six turn retrievals, on NEW dicts that never alias the input.
  (d) battery_records / frac_origin / origin_counts arithmetic on a
      hand-built fixture (origin turn of store index i is i+1).

Determinism note: the fake embed hashes each token with md5 (stable across
processes, unlike builtin hash()), so two identical build sequences produce
bit-identical vector stores and the expected top-k is reproducible.
"""
from __future__ import annotations

import copy
import hashlib
import sys

import memory_g3
from memory_g3 import VectorStore
from contract_g2 import PROBES, PRESSURE
from contract_g35 import SINK_QUERY_PREFIX
from provenance_g35 import (battery_records, classify, frac_origin,
                            origin_counts)


_FAILS = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    if not cond:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# Deterministic fake embed: md5 of each whitespace token picks a dimension and
# a weight. Pure, process-stable, zero daemon calls.
# ---------------------------------------------------------------------------
_DIM = 16


def _hash_embed(text):
    """Hash-based fixed-dim embedding: for each lowercased token, md5 byte 0
    selects the dimension and byte 1 the added weight. Deterministic across
    processes (md5, not the salted builtin hash), so the two equivalence
    passes build bit-identical vector stores."""
    vec = [0.0] * _DIM
    for tok in text.lower().split():
        h = hashlib.md5(tok.encode("utf-8")).digest()
        vec[h[0] % _DIM] += 1.0 + h[1] / 255.0
    return vec


def _make_counting_embed(box):
    """Wrap _hash_embed so `box[0]` counts invocations (proves the sink adds
    no embed call: append/retrieve embed counts are identical off vs on)."""
    def fn(text):
        box[0] += 1
        return _hash_embed(text)
    return fn


# ---------------------------------------------------------------------------
# The FIXED build + retrieve sequence shared by both equivalence passes.
# 10 entries (5 permissive + 5 caution turns, mirroring the hyst arm), then
# 8 retrieves: two verbatim PROBES texts + six verbatim turn texts, mixed k.
# ---------------------------------------------------------------------------
_TURNS = ([(PRESSURE["permissive"][i], "soft reply %d" % i) for i in range(5)]
          + [(PRESSURE["caution"][i], "firm reply %d" % i) for i in range(5)])

_QUERIES = [
    (PROBES[0]["text"], 4),            # probe: deploy-prod (>80 chars)
    (PRESSURE["neutral"][0], 4),       # turn
    (PRESSURE["permissive"][1], 4),    # turn
    (PROBES[3]["text"], 4),            # probe: hardcode-key (>80 chars)
    (PRESSURE["caution"][2], 2),       # turn, k=2
    (PRESSURE["neutral"][3], 3),       # turn, k=3
    (PRESSURE["permissive"][0], 4),    # turn (equals an entry's user text)
    (PRESSURE["caution"][4], 4),       # turn
]
_PROBE_POSITIONS = {0: "deploy-prod", 3: "hardcode-key"}


def _build_store(embed_box):
    store = VectorStore(client=None, counter={"n_embed": 0},
                        _embed=_make_counting_embed(embed_box))
    for user_msg, reply in _TURNS:
        store.append_turn(user_msg, reply)
    return store


def _run_retrieves(store):
    return [store.retrieve(q, k=k) for q, k in _QUERIES]


# ---------------------------------------------------------------------------
# (a) sink-off vs sink-on equivalence + frozen record shape
# ---------------------------------------------------------------------------
def test_sink_equivalence():
    print("(a) sink OFF vs ON: bit-identical returns/snapshot; record shape")
    _check("RETRIEVAL_SINK defaults to None (fresh import)",
           memory_g3.RETRIEVAL_SINK is None)

    # -- pass 1: sink OFF --
    box_off = [0]
    s1 = _build_store(box_off)
    returns_off = _run_retrieves(s1)
    snap_off = s1.snapshot()

    # -- pass 2: sink ON, identical build + identical retrieves --
    memory_g3.RETRIEVAL_SINK = []
    box_on = [0]
    s2 = _build_store(box_on)
    returns_on = _run_retrieves(s2)
    snap_on = s2.snapshot()
    sink = memory_g3.RETRIEVAL_SINK

    _check("returns bit-identical (8 retrieves)", returns_on == returns_off)
    _check("snapshots bit-identical", snap_on == snap_off)
    _check("embed count identical off vs on (sink adds no embed)",
           box_off[0] == box_on[0] == len(_TURNS) + len(_QUERIES),
           "off=%d on=%d" % (box_off[0], box_on[0]))
    _check("sink captured one record per retrieve", len(sink) == len(_QUERIES),
           "len=%d" % len(sink))

    # -- frozen record shape, cross-checked field by field ----------------
    q_ok = sz_ok = k_ok = top_ok = map_ok = cos_ok = True
    detail = ""
    for rec, (query, k), ret in zip(sink, _QUERIES, returns_on):
        if rec["query"] != query[:SINK_QUERY_PREFIX]:
            q_ok = False
            detail = detail or "query=%r" % (rec["query"],)
        if rec["store_size"] != len(_TURNS):
            sz_ok = False
            detail = detail or "store_size=%r" % (rec["store_size"],)
        if rec["k"] != k:
            k_ok = False
            detail = detail or "k=%r != %r" % (rec["k"], k)
        top = rec["top"]
        if (not isinstance(top, list)
                or not all(isinstance(i, int) for i in top)
                or top != sorted(top) or len(set(top)) != len(top)
                or len(top) != k):
            top_ok = False
            detail = detail or "top=%r" % (top,)
        elif [s2.entries[i] for i in top] != ret:
            map_ok = False
            detail = detail or "top does not map to the returned entries"
        else:
            qvec = _hash_embed(query)
            exp_cos = [memory_g3._cosine(qvec, s2.vectors[i]) for i in top]
            if rec["cos"] != exp_cos or len(rec["cos"]) != len(top):
                cos_ok = False
                detail = detail or "cos mismatch for query %r" % (query[:20],)
    _check("record query == query[:80]", q_ok, detail if not q_ok else "")
    _check("record store_size == query-time size (10)", sz_ok)
    _check("record k == the k passed", k_ok)
    _check("record top: sorted unique ints, len == k (oldest-first)", top_ok,
           detail if not top_ok else "")
    _check("record top maps exactly onto the returned entries", map_ok)
    _check("record cos: bit-identical to independent recomputation, "
           "parallel to top", cos_ok, detail if not cos_ok else "")
    # the two probe queries exceed 80 chars -> stored prefix is exactly 80
    _check("probe queries truncated to exactly SINK_QUERY_PREFIX chars",
           all(len(sink[i]["query"]) == SINK_QUERY_PREFIX
               for i in _PROBE_POSITIONS))

    # pristine deep-ish copies for the classification test (before any
    # mutation the isolation test performs on the live sink records).
    captured = [dict(r, top=list(r["top"]), cos=list(r["cos"])) for r in sink]
    return {"sink": sink, "s2": s2, "snap_on": snap_on,
            "returns_on": returns_on, "captured": captured}


# ---------------------------------------------------------------------------
# (b) sink isolation: query-time size, restore/reset, no aliasing, off-switch
# ---------------------------------------------------------------------------
def test_sink_isolation(ctx):
    print("(b) sink isolation: restore/reset untouched, records never alias")
    sink = ctx["sink"]
    s2 = ctx["s2"]
    snap = ctx["snap_on"]
    returns_on = ctx["returns_on"]
    captured = ctx["captured"]

    # store_size is the QUERY-TIME size (append one, retrieve -> 11)
    s2.append_turn("one extra user turn", "one extra reply")
    extra = s2.retrieve(PRESSURE["neutral"][1], k=4)
    _check("record store_size tracks query-time size (11 after extra append)",
           len(sink) == 9 and sink[-1]["store_size"] == 11 and len(extra) == 4,
           "len(sink)=%d last_size=%r" % (len(sink),
                                          sink[-1]["store_size"]))

    # restore: store back to the probed state; sink untouched
    before = copy.deepcopy(sink)
    s2.restore(snap)
    _check("restore leaves the sink untouched (content bit-identical)",
           sink == before)
    _check("restore returned the store to the probed state",
           s2.snapshot() == snap)

    # aliasing: mutating a sink record's lists must not touch the store
    q0, k0 = _QUERIES[0]
    sink[0]["top"][0] = -999
    sink[0]["cos"][0] = 123.456
    re_ret = s2.retrieve(q0, k=k0)          # appends a fresh record
    _check("store unaffected by record mutation (snapshot bit-identical)",
           s2.snapshot() == snap)
    _check("re-retrieve after record mutation is bit-identical",
           re_ret == returns_on[0])
    _check("fresh record rebuilt pristine (independent of the mutation)",
           sink[-1]["top"] == captured[0]["top"]
           and sink[-1]["cos"] == captured[0]["cos"])

    # reset: sink untouched; empty-store retrieve logs NOTHING
    n = len(sink)
    s2.reset()
    empty = s2.retrieve("anything at all", k=4)
    _check("reset leaves the sink untouched", len(sink) == n)
    _check("empty-store retrieve returns [] and logs no record",
           empty == [] and len(sink) == n)

    # off-switch: None restores the silent default path
    memory_g3.RETRIEVAL_SINK = None
    box = [0]
    s3 = _build_store(box)
    off_ret = s3.retrieve(q0, k=k0)
    _check("sink OFF again: retrieval unchanged, nothing logged",
           off_ret == returns_on[0] and memory_g3.RETRIEVAL_SINK is None)


# ---------------------------------------------------------------------------
# (c) classification of the captured sink records
# ---------------------------------------------------------------------------
def test_classification(ctx):
    print("(c) provenance_g35.classify on the captured sink records")
    captured = ctx["captured"]
    labeled = classify(captured)

    _check("classify returns one NEW dict per record",
           len(labeled) == len(captured)
           and all(a is not b for a, b in zip(labeled, captured)))
    _check("classify never mutates the input (no kind key leaks back)",
           all("kind" not in r and "probe_id" not in r for r in captured))
    _check("classify output lists never alias the input lists",
           all(a["top"] is not b["top"] and a["cos"] is not b["cos"]
               for a, b in zip(labeled, captured)))

    kinds_ok = True
    detail = ""
    for i, rec in enumerate(labeled):
        if i in _PROBE_POSITIONS:
            if rec["kind"] != "probe" or rec["probe_id"] != _PROBE_POSITIONS[i]:
                kinds_ok = False
                detail = detail or "i=%d kind=%r id=%r" % (i, rec["kind"],
                                                           rec["probe_id"])
        else:
            if rec["kind"] != "turn" or rec["probe_id"] is not None:
                kinds_ok = False
                detail = detail or "i=%d kind=%r id=%r" % (i, rec["kind"],
                                                           rec["probe_id"])
    _check("2 probe records (deploy-prod, hardcode-key) + 6 turn records",
           kinds_ok, detail)

    # exactness: a 79-char near-miss of a probe prefix is a TURN
    near_miss = {"store_size": 10,
                 "query": PROBES[0]["text"][:SINK_QUERY_PREFIX - 1],
                 "top": [0], "cos": [1.0], "k": 4}
    lab = classify([near_miss])[0]
    _check("79-char probe-prefix near-miss classifies as turn",
           lab["kind"] == "turn" and lab["probe_id"] is None)


# ---------------------------------------------------------------------------
# (d) battery_records / frac_origin / origin_counts arithmetic (fixture)
# ---------------------------------------------------------------------------
def test_fixture_arithmetic():
    print("(d) battery_records / frac_origin / origin_counts (hand fixture)")
    p = SINK_QUERY_PREFIX
    # 4 records: probe@15, turn@15, probe@35, probe@35 -- store indices chosen
    # so the pooled origin turns are hand-checkable (origin turn = index + 1).
    fx = [
        {"store_size": 15, "query": PROBES[0]["text"][:p],
         "top": [0, 1, 2, 3], "cos": [.9, .8, .7, .6], "k": 4},
        {"store_size": 15, "query": PRESSURE["neutral"][0],
         "top": [0, 1], "cos": [.9, .8], "k": 4},
        {"store_size": 35, "query": PROBES[1]["text"][:p],
         "top": [14, 15, 33, 34], "cos": [.9, .8, .7, .6], "k": 4},
        {"store_size": 35, "query": PROBES[2]["text"][:p],
         "top": [0, 14, 15, 16], "cos": [.9, .8, .7, .6], "k": 4},
    ]

    br15 = battery_records(fx, 15)
    br35 = battery_records(fx, 35)
    _check("battery_records(15): the 1 probe record at t=15 (turn excluded)",
           len(br15) == 1 and br15[0]["probe_id"] == PROBES[0]["id"])
    _check("battery_records(35): the 2 probe records at t=35",
           len(br35) == 2 and [r["probe_id"] for r in br35] ==
           [PROBES[1]["id"], PROBES[2]["id"]])
    _check("battery_records on unmatched t -> []",
           battery_records(fx, 20) == [])
    _check("battery_records accepts pre-classified input (same selection)",
           [r["probe_id"] for r in battery_records(classify(fx), 35)] ==
           [r["probe_id"] for r in br35])

    # br35 pooled top = [14,15,33,34, 0,14,15,16]
    # -> origin turns   [15,16,34,35, 1,15,16,17]
    # in [1,15]: 15,1,15            -> 3/8 (soft_frac_ret shape)
    # in [16,35]: 16,34,35,16,17    -> 5/8 (firm_frac_ret shape)
    _check("frac_origin(br35, 1, 15) == 3/8",
           frac_origin(br35, 1, 15) == 3 / 8,
           "got=%r" % (frac_origin(br35, 1, 15),))
    _check("frac_origin(br35, 16, 35) == 5/8",
           frac_origin(br35, 16, 35) == 5 / 8)
    _check("frac_origin complements sum to 1 over a partition",
           frac_origin(br35, 1, 15) + frac_origin(br35, 16, 35) == 1.0)
    _check("frac_origin(br35, 1, 35) == 1.0 (all origins in range)",
           frac_origin(br35, 1, 35) == 1.0)
    _check("frac_origin(br35, 18, 33) == 0.0 (no origins in range)",
           frac_origin(br35, 18, 33) == 0.0)
    _check("frac_origin([]) is None (not 0.0)",
           frac_origin([], 1, 15) is None)
    _check("frac_origin with only empty top lists is None",
           frac_origin([{"store_size": 5, "query": "x", "top": [],
                         "cos": [], "k": 4}], 1, 15) is None)

    _check("origin_counts(br35) == {15:2, 16:2, 34:1, 35:1, 1:1, 17:1}",
           origin_counts(br35) ==
           {15: 2, 16: 2, 34: 1, 35: 1, 1: 1, 17: 1},
           "got=%r" % (origin_counts(br35),))
    _check("origin_counts totals == pooled item count (8)",
           sum(origin_counts(br35).values()) == 8)
    _check("origin_counts([]) == {}", origin_counts([]) == {})


def main():
    print("=" * 66)
    print("selftest_a35 -- DRY proofs: RETRIEVAL_SINK + provenance_g35 "
          "(0 daemon calls)")
    print("=" * 66)
    ctx = test_sink_equivalence()
    test_sink_isolation(ctx)
    test_classification(ctx)
    test_fixture_arithmetic()
    print("=" * 66)
    if _FAILS:
        print("RESULT: FAIL (%d failing checks): %s" % (len(_FAILS), _FAILS))
        sys.exit(1)
    print("RESULT: PASS (all checks)")
    sys.exit(0)


if __name__ == "__main__":
    main()
