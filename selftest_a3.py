"""selftest_a3.py -- G3 subagent A: DRY self-tests (ZERO daemon calls).

Run:  python selftest_a3.py

Covers memory_g3.py end to end with injected fakes -- no ollama import, no
network, no daemon:

  (a) both stores' snapshot/restore/reset round-trip BIT-IDENTICALLY, including
      R3 vectors (with fake injected embeddings);
  (b) R1 retrieve returns the k most recent entries, oldest first (query ignored);
  (c) R3 cosine top-k with injected fake vectors picks the right entries, and the
      recency tiebreak works; oldest-first output order verified;
  (d) entry format + 400-char reply cap ("[user] " / " [you] " substrings);
  (e) the G3 agent_turn adapter calls respond THEN append_turn, and SKIPS append
      on memoryless (injected fake respond -- no ollama, no daemon), called
      exactly as run_life calls it (model=, flavor= kwargs present);
  (f) the backend dict shape drives a 5-turn experiments_g2.run_life with the
      stub backend (make_backend_g3 use_stub=True).

Determinism note: R3 tests inject a deterministic bag-of-words embed (normalized
word-count over a fixed vocabulary), so cosine similarity tracks lexical overlap
and the expected top-k is hand-verifiable with no daemon.
"""
from __future__ import annotations

import sys

import memory_g3
from memory_g3 import RawLogStore, VectorStore, format_entry, make_backend_g3


_FAILS = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    if not cond:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# Deterministic fake embed: bag-of-words over a fixed vocabulary, normalized.
# Cosine similarity then tracks lexical overlap -> hand-verifiable top-k.
# ---------------------------------------------------------------------------
_VOCAB = ["apple", "banana", "cherry", "deploy", "backup", "config",
          "user", "you", "the", "a", "to", "and"]
_VOCAB_INDEX = {w: i for i, w in enumerate(_VOCAB)}


def _fake_embed(text):
    """Normalized word-count vector over _VOCAB (words outside vocab ignored).
    Deterministic, pure, zero daemon calls. NOTE: an injected _embed seam does
    NOT touch the counter (only the store's real _embed_daemon counts n_embed);
    the counting path is tested separately in test_embed_counting via a fake
    client that owns an .embed method."""
    import re
    vec = [0.0] * len(_VOCAB)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        j = _VOCAB_INDEX.get(w)
        if j is not None:
            vec[j] += 1.0
    return vec


class _FakeEmbedClient:
    """A fake ollama client exposing ONLY .embed, returning an object with an
    .embeddings attribute (list-of-vectors, matching EmbedResponse). Lets the
    REAL VectorStore._embed_daemon run -- and thus increment counter['n_embed']
    -- with zero daemon/network calls."""

    class _Resp:
        def __init__(self, vec):
            self.embeddings = [vec]

    def __init__(self):
        self.n_embed_calls = 0

    def embed(self, model, input):     # noqa: A002 -- mirror ollama's kwarg name
        self.n_embed_calls += 1
        return self._Resp(_fake_embed(input))


# ---------------------------------------------------------------------------
# (a) snapshot / restore / reset round-trip (both stores)
# ---------------------------------------------------------------------------
def test_roundtrip():
    print("(a) snapshot/restore/reset round-trip (R1 + R3)")

    # -- R1 --
    r1 = RawLogStore()
    r1.append_turn("hello", "world reply")
    r1.append_turn("second", "another reply")
    snap = r1.snapshot()
    before = list(r1.entries)
    # mutate, then restore
    r1.append_turn("third", "x")
    r1.restore(snap)
    _check("R1 restore is bit-identical", r1.entries == before,
           "entries=%r" % (r1.entries,))
    # snapshot independence: mutating the store must not change the snapshot
    r1.append_turn("fourth", "y")
    _check("R1 snapshot independent of later writes", snap == before)
    r1.reset()
    _check("R1 reset clears entries", r1.entries == [])

    # -- R3 (fake injected embeddings) --
    r3 = VectorStore(client=None, counter={"n_embed": 0}, _embed=_fake_embed)
    r3.append_turn("apple banana", "cherry deploy")
    r3.append_turn("backup config", "user you")
    snap3 = r3.snapshot()
    before_entries = list(r3.entries)
    before_vectors = [list(v) for v in r3.vectors]
    # mutate then restore
    r3.append_turn("more apple", "more banana")
    r3.restore(snap3)
    _check("R3 restore entries bit-identical", r3.entries == before_entries)
    _check("R3 restore vectors bit-identical", r3.vectors == before_vectors,
           "vectors match=%s" % (r3.vectors == before_vectors))
    # snapshot deep-independence: mutating a restored vector must not touch snap
    r3.vectors[0][0] = 999.0
    snap_entries, snap_vectors = snap3
    _check("R3 snapshot vectors deep-independent", snap_vectors[0][0] != 999.0,
           "snap[0][0]=%r" % (snap_vectors[0][0],))
    r3.reset()
    _check("R3 reset clears entries+vectors",
           r3.entries == [] and r3.vectors == [])


# ---------------------------------------------------------------------------
# (b) R1 retrieve: k most recent, oldest first, query ignored
# ---------------------------------------------------------------------------
def test_r1_retrieve():
    print("(b) R1 retrieve: k most recent, oldest first, query ignored")
    r1 = RawLogStore()
    _check("R1 empty retrieve -> []", r1.retrieve("anything", k=4) == [])
    for i in range(6):
        r1.append_turn("msg%d" % i, "reply%d" % i)
    got = r1.retrieve("irrelevant query text", k=4)
    # entries are e0..e5; k=4 most recent = e2,e3,e4,e5 oldest-first
    expected = [format_entry("msg%d" % i, "reply%d" % i) for i in (2, 3, 4, 5)]
    _check("R1 returns k most recent oldest-first", got == expected,
           "got tail idx=%s" % ([e.split()[1] for e in got]))
    # query is ignored: a query matching an OLD entry does not surface it
    got2 = r1.retrieve("msg0", k=2)
    expected2 = [format_entry("msg4", "reply4"), format_entry("msg5", "reply5")]
    _check("R1 ignores query (still most-recent)", got2 == expected2)
    # k larger than store -> all entries oldest-first
    small = RawLogStore()
    small.append_turn("only", "one")
    _check("R1 k>len returns all", small.retrieve("q", k=4) ==
           [format_entry("only", "one")])


# ---------------------------------------------------------------------------
# (c) R3 cosine top-k + recency tiebreak, oldest-first output
# ---------------------------------------------------------------------------
def test_r3_retrieve():
    print("(c) R3 cosine top-k, recency tiebreak, oldest-first")
    r3 = VectorStore(client=None, counter={"n_embed": 0}, _embed=_fake_embed)
    _check("R3 empty retrieve -> [] (no embed)", r3.retrieve("apple", k=4) == [])

    # four entries with controlled vocabulary overlap:
    r3.append_turn("apple apple", "fruit")       # e0: apple x2
    r3.append_turn("banana", "yellow")           # e1: banana x1
    r3.append_turn("cherry deploy", "red")       # e2: cherry, deploy
    r3.append_turn("apple cherry", "mix")        # e3: apple, cherry

    # query "apple" -> highest cosine on entries containing apple (e0, e3),
    # top-2 should be {e0, e3} returned oldest-first (e0 then e3).
    got = r3.retrieve("apple", k=2)
    exp = [r3.entries[0], r3.entries[3]]
    _check("R3 top-2 for 'apple' = {e0,e3} oldest-first", got == exp,
           "got=%r" % ([e[:16] for e in got]))

    # query "cherry" -> e2 and e3 both contain cherry once; e2 also has deploy
    # (dilutes cosine), e3 has apple (dilutes). Both nonzero; pull top-2.
    got_c = r3.retrieve("cherry", k=2)
    _check("R3 top-2 for 'cherry' are the two cherry entries oldest-first",
           got_c == [r3.entries[2], r3.entries[3]],
           "got=%r" % ([e[:16] for e in got_c]))

    # recency tiebreak: two identical-vector entries; a query equally similar to
    # both. With k=1 the MORE RECENT one must win the tie (index desc), and the
    # returned single item is that recent entry.
    tie = VectorStore(client=None, counter={"n_embed": 0}, _embed=_fake_embed)
    tie.append_turn("banana", "old")     # e0
    tie.append_turn("apple", "noise")    # e1 (different vector)
    tie.append_turn("banana", "new")     # e2 (same vector as e0)
    got_tie = tie.retrieve("banana", k=1)
    _check("R3 recency tiebreak picks the more recent equal-sim entry",
           got_tie == [tie.entries[2]],
           "got=%r" % ([e[:20] for e in got_tie]))


# ---------------------------------------------------------------------------
# (d) entry format + 400-char cap
# ---------------------------------------------------------------------------
def test_entry_format():
    print("(d) entry format + 400-char reply cap")
    e = format_entry("do X", "sure, done")
    _check("entry contains '[user] '", "[user] " in e)
    _check("entry contains ' [you] '", " [you] " in e)
    _check("entry exact shape", e == "[user] do X [you] sure, done", repr(e))
    long_reply = "z" * 900
    e2 = format_entry("q", long_reply)
    # the reply slice is capped at 400 chars
    reply_part = e2.split(" [you] ", 1)[1]
    _check("reply capped at 400 chars", len(reply_part) == 400,
           "len=%d" % len(reply_part))
    # a user message is NOT capped (only the reply slice is): the message text
    # survives verbatim between the tags regardless of length.
    e3 = format_entry("Z" * 500, "r")
    user_part = e3.split(" [you] ", 1)[0][len("[user] "):]
    _check("user message not capped", user_part == "Z" * 500,
           "user_len=%d" % len(user_part))


# ---------------------------------------------------------------------------
# (d2) R3 embed counting: the REAL _embed_daemon counts n_embed via a fake
# client (zero daemon). Injected _embed seams (used elsewhere) do NOT count --
# only the store's daemon path bumps the ledger.
# ---------------------------------------------------------------------------
def test_embed_counting():
    print("(d2) R3 embed counting: _embed_daemon bumps counter['n_embed']")
    fake_client = _FakeEmbedClient()
    counter = {"n_calls": 0, "n_embed": 0}
    # NOTE: no _embed injection -> the store uses its real _embed_daemon, which
    # calls fake_client.embed and increments counter['n_embed'].
    r3 = VectorStore(client=fake_client, counter=counter)

    # empty retrieve must NOT embed (short-circuit before embedding)
    _check("empty retrieve short-circuits (no embed)",
           r3.retrieve("apple", k=4) == [] and counter["n_embed"] == 0)

    r3.append_turn("apple banana", "cherry")     # 1 embed
    r3.append_turn("deploy backup", "config")    # 1 embed
    _check("append embeds once each (n_embed=2)", counter["n_embed"] == 2,
           "n_embed=%d" % counter["n_embed"])

    r3.retrieve("apple", k=2)                     # 1 embed (query)
    _check("retrieve embeds the query (n_embed=3)", counter["n_embed"] == 3,
           "n_embed=%d" % counter["n_embed"])

    # embeds are NEVER counted as chat calls (separate ledger key)
    _check("embeds never touch n_calls", counter["n_calls"] == 0,
           "n_calls=%d" % counter["n_calls"])
    _check("fake client embed call count matches ledger",
           fake_client.n_embed_calls == 3,
           "client=%d ledger=%d" % (fake_client.n_embed_calls,
                                    counter["n_embed"]))

    # snapshot/restore make NO embed calls (data-only)
    before = counter["n_embed"]
    snap = r3.snapshot()
    r3.restore(snap)
    _check("snapshot/restore make no embed calls",
           counter["n_embed"] == before)


# ---------------------------------------------------------------------------
# (e) G3 agent_turn adapter: respond then append; skip append on memoryless
# ---------------------------------------------------------------------------
def test_agent_turn_adapter():
    print("(e) G3 agent_turn: respond->append_turn; skip append on memoryless")
    # build the REAL backend but never touch the daemon: we inject _respond and
    # only exercise the adapter (make_store/agent_turn), not run_battery.
    counter = {"n_calls": 0, "n_embed": 0}
    backend = memory_g3._real_backend_g3("R1", counter)
    agent_turn = backend["agent_turn"]
    make_store = backend["make_store"]

    calls = {"respond": 0}

    def fake_respond(user_msg, store, *, client, model):
        calls["respond"] += 1
        return "FAKE REPLY to " + user_msg

    # memory path: called EXACTLY as run_life calls it (model=, flavor= present)
    store = make_store()
    tr = agent_turn("hello there", store, 0, client=None, model="ignored-tag",
                    memoryless=False, flavor="neutral", _respond=fake_respond)
    _check("adapter returned a reply", tr.reply == "FAKE REPLY to hello there",
           "reply=%r" % tr.reply)
    _check("adapter note is None (no note call on G3 rungs)", tr.note is None)
    _check("adapter called respond once", calls["respond"] == 1)
    _check("adapter appended one entry", len(store.entries) == 1)
    _check("appended entry has both segments",
           "[user] " in store.entries[0] and " [you] " in store.entries[0])

    # memoryless path: reset before respond, append NOTHING
    store2 = make_store()
    store2.append_turn("pre-existing", "junk")     # would be wiped by reset
    tr2 = agent_turn("q", store2, 1, client=None, model="tag", memoryless=True,
                     flavor="neutral", _respond=fake_respond)
    _check("memoryless returns reply", tr2.reply == "FAKE REPLY to q")
    _check("memoryless store empty after turn (reset + no append)",
           store2.entries == [], "entries=%r" % (store2.entries,))
    _check("memoryless respond still called", calls["respond"] == 2)

    # signature robustness: the adapter must accept model= and flavor= (run_life
    # always passes them) AND default them (defensive).
    store3 = make_store()
    try:
        agent_turn("x", store3, 2, client=None, _respond=fake_respond)
        sig_ok = True
    except TypeError as exc:
        sig_ok = False
        _check("adapter tolerates omitted model/flavor", False, repr(exc))
    if sig_ok:
        _check("adapter tolerates omitted model/flavor", True)


# ---------------------------------------------------------------------------
# (f) backend dict shape drives a 5-turn run_life (stub backend)
# ---------------------------------------------------------------------------
def test_backend_shape_runlife():
    print("(f) make_backend_g3 stub backend drives run_life (5 turns)")
    import experiments_g2

    counter = {"n_calls": 0, "n_embed": 0}
    backend = make_backend_g3("R1", use_stub=True, counter=counter)
    for key in ("client", "make_store", "agent_turn", "run_battery", "kind"):
        _check("backend has %r" % key, key in backend)

    specs = [("neutral", "text %d" % i) for i in range(5)]
    try:
        out = experiments_g2.run_life(
            specs, backend=backend, memoryless=False, probe_at=[3, 5],
            seed=1, counter=counter, ceiling=10_000, label="selftest_a3")
        ok = ("probes" in out and len(out["probes"]) == 2
              and "notes_count" in out)
        _check("run_life completed with 2 probes", ok,
               "probes=%d calls=%d" % (len(out.get("probes", [])),
                                       counter["n_calls"]))
    except Exception as exc:  # noqa: BLE001
        _check("run_life completed without error", False, repr(exc))


def main():
    print("=" * 62)
    print("selftest_a3 -- DRY self-tests for memory_g3 (0 daemon calls)")
    print("=" * 62)
    test_roundtrip()
    test_r1_retrieve()
    test_r3_retrieve()
    test_entry_format()
    test_embed_counting()
    test_agent_turn_adapter()
    test_backend_shape_runlife()
    print("=" * 62)
    if _FAILS:
        print("RESULT: FAIL (%d failing checks): %s" % (len(_FAILS), _FAILS))
        sys.exit(1)
    print("RESULT: PASS (all checks)")
    sys.exit(0)


if __name__ == "__main__":
    main()
