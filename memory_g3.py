"""memory_g3.py -- G3 subagent A: the two NEW memory backends (rungs R1, R3).

The G3 ladder runs a family of memory architectures through the IDENTICAL
path-dependence protocol (G2's Exp-2, battery25) and asks which architectural
property produces attractors. Only two rungs are new:

  R1 raw log   (RawLogStore)  -- append-only verbatim log; retrieve the k most
                                 recent entries, oldest first; NO summarization,
                                 NO self-rewrite, NO perception-coupling.
  R3 vector    (VectorStore)  -- embed each entry once at append time
                                 (EMBED_MODEL, local, counted separately as
                                 "n_embed"); retrieve top-k by cosine similarity
                                 to the embedded query, ties by recency, oldest
                                 first. The "sophisticated memory" the reviewer
                                 demands, WITHOUT perception-coupling -- the
                                 sharp test.

Both stores are drop-ins for the memory component of the existing agent loop:
they implement the surface agent.respond and battery25 need
(.summary, .retrieve, .snapshot/.restore/.reset) plus the frozen G3
.append_turn(user_msg, reply). agent.respond and AGENT_SYSTEM are reused
VERBATIM -- the identity prompt is part of the instrument, and entries appear
under the same "past notes" block.

Frozen discipline (contract_g3.py):
  * .summary is ALWAYS "" -- there is no summarizer on these rungs (that is the
    architecture under test: persistence WITHOUT self-authored compression).
  * Entry format: one entry per turn, "[user] <msg> [you] <reply>", with the
    reply slice capped at 400 chars.
  * NO note-writing LLM call, NO summary call on these rungs. The G3 agent_turn
    does exactly: reply = agent.respond(...); store.append_turn(user_msg, reply).
  * R3 embedding daemon calls are counted in a SEPARATE counter key "n_embed",
    never against chat ceilings.
  * snapshot/restore are DATA-ONLY deep copies (entries + vectors). They must
    NOT deepcopy the ollama client or the counter -- battery25 snapshots on
    every probe and a client is not copyable, while forking the counter would
    corrupt the embed ledger.

Contract: contract_g3.py (frozen) + contract_g2.py (reused). Do not diverge.
"""
from __future__ import annotations

import copy
import math

from contract_g2 import MODEL, RETRIEVE_K
from contract_g3 import EMBED_MODEL

# Frozen entry format (contract_g3.py "New-rung memory discipline").
_REPLY_CAP = 400
_USER_TAG = "[user] "
_YOU_TAG = " [you] "


def format_entry(user_msg: str, reply: str) -> str:
    """The frozen per-turn entry string: "[user] <msg> [you] <reply>" with the
    reply slice capped at 400 chars. Used by BOTH rungs so the stored text is
    architecture-independent (only the RETRIEVAL differs across rungs)."""
    return _USER_TAG + user_msg + _YOU_TAG + reply[:_REPLY_CAP]


# ---------------------------------------------------------------------------
# R1 -- raw log store (append-only; recent-N retrieval; no coupling)
# ---------------------------------------------------------------------------
class RawLogStore:
    """Rung R1: an append-only verbatim log.

    retrieve(query, k) returns the k most recent entries, oldest first, and
    IGNORES the query (a pure log has no relevance model). No summarization, no
    rewrite -- append-only persistence, the minimal non-trivial memory. .summary
    is always "" (no summarizer on this rung).
    """

    def __init__(self) -> None:
        self.entries: list = []      # list[str], verbatim, append-only per life

    @property
    def summary(self) -> str:
        # ALWAYS "" -- no summarizer on this rung (frozen).
        return ""

    # -- writing --------------------------------------------------------
    def append_turn(self, user_msg: str, reply: str) -> None:
        """Append one frozen-format entry for this turn."""
        self.entries.append(format_entry(user_msg, reply))

    # -- retrieval (recent-N, oldest first; query ignored) --------------
    def retrieve(self, query: str, k: int = RETRIEVE_K) -> list:
        """The k most recent entries, returned oldest first. Query is ignored
        (pure log). Pure read: never mutates the store. Returns [] when empty."""
        if not self.entries:
            return []
        # last k entries, already in chronological (oldest-first) order.
        return list(self.entries[-k:])

    # -- read-only probing support (measuring must not mutate) ----------
    def snapshot(self):
        """Deep, data-only copy: an independent list of the entry strings.
        Returned as an opaque token battery25 treats as data (never a live
        store), so nothing done through it can touch this store."""
        return list(self.entries)   # strings are immutable; a fresh list is deep

    def restore(self, snap) -> None:
        """Restore entries from a snapshot() result."""
        self.entries = list(snap)

    def reset(self) -> None:
        """Clear all memory (used by the memoryless control, every turn)."""
        self.entries = []


# ---------------------------------------------------------------------------
# R3 -- vector store (embed each entry; top-k cosine retrieval; no coupling)
# ---------------------------------------------------------------------------
def _cosine(a: list, b: list) -> float:
    """Cosine similarity of two equal-length float vectors. 0.0 if either norm
    is 0 (defensive -- a real embedding is never all-zero)."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class VectorStore:
    """Rung R3: a vector memory (embed + top-k similarity retrieval).

    At append time the FULL entry text is embedded once (EMBED_MODEL) and stored
    alongside the entry. retrieve embeds the QUERY, ranks stored entries by
    cosine similarity (ties by recency), and returns the top-k OLDEST FIRST.
    NO summarization, NO self-rewrite, NO scalar state, NO perception-coupling
    (the retrieved memories only feed retrieval/expression, never the appraisal
    of new input). This is the "sophisticated memory" the reviewer demands.

    Embedding is a daemon call: it is counted in a SEPARATE counter key
    "n_embed", NOT against the chat ceilings. The store closes over
    (client, counter) at construction so append_turn/retrieve can embed and
    count without the caller threading a client (the G3 agent_turn calls
    append_turn(user_msg, reply) with no client, mirroring MemoryStore.write).

    A private `_embed` seam (defaults to the real daemon embed) lets dry tests
    inject deterministic fake embeddings with zero daemon calls.
    """

    def __init__(self, client=None, counter=None, *, embed_model: str = EMBED_MODEL,
                 _embed=None) -> None:
        self.entries: list = []      # list[str], verbatim, append-only per life
        self.vectors: list = []      # list[list[float]], parallel to entries
        self._client = client
        self._counter = counter
        self._embed_model = embed_model
        # injection seam for dry tests; None -> the real counted daemon embed.
        self._embed_fn = _embed if _embed is not None else self._embed_daemon

    @property
    def summary(self) -> str:
        # ALWAYS "" -- no summarizer on this rung (frozen).
        return ""

    # -- embedding (the ONE daemon touch; counted as n_embed, not n_calls) --
    def _embed_daemon(self, text: str) -> list:
        """Embed one string via EMBED_MODEL and return a plain list[float].

        Counts the call in counter["n_embed"] (CountingClient does NOT wrap
        .embed, so the store must count it). ollama Client.embed returns an
        EmbedResponse whose .embeddings is a LIST of vectors (one per input);
        with a single input the per-input vector is embeddings[0]. Stored as a
        plain list of floats (no numpy) so snapshot/restore equality is exact.
        """
        if self._counter is not None:
            self._counter["n_embed"] = self._counter.get("n_embed", 0) + 1
        resp = self._client.embed(model=self._embed_model, input=text)
        # EmbedResponse.embeddings: list-of-vectors even for a single input.
        vec = resp.embeddings[0]
        return [float(x) for x in vec]

    # -- writing --------------------------------------------------------
    def append_turn(self, user_msg: str, reply: str) -> None:
        """Append one frozen-format entry AND embed its full text once."""
        entry = format_entry(user_msg, reply)
        vec = self._embed_fn(entry)
        self.entries.append(entry)
        self.vectors.append([float(x) for x in vec])

    # -- retrieval (top-k cosine; recency tiebreak; oldest first) -------
    def retrieve(self, query: str, k: int = RETRIEVE_K) -> list:
        """Top-k entries by cosine similarity between embed(query) and each
        stored per-entry vector; ties broken by recency (later entry wins);
        the selected k returned OLDEST FIRST.

        Pure read: never appends, never caches (H5 / restore integrity depend on
        this). Short-circuits to [] on an empty store BEFORE embedding, so an
        empty store needs no daemon call and is architecture-independent.
        """
        if not self.entries:
            return []
        qvec = self._embed_fn(query)
        # rank by (cosine desc, index desc) so recent entries win similarity
        # ties; take the top-k, then re-sort those k by index ASC (oldest first).
        ranked = sorted(
            range(len(self.entries)),
            key=lambda i: (_cosine(qvec, self.vectors[i]), i),
            reverse=True,
        )
        top = sorted(ranked[:k])            # indices, oldest-first
        return [self.entries[i] for i in top]

    # -- read-only probing support (deep-copy entries AND vectors) ------
    def snapshot(self):
        """Deep, DATA-ONLY copy: (entries, vectors) with vectors deep-copied.

        Deliberately does NOT copy the client or counter (battery25 snapshots on
        every probe; the ollama client is not deep-copyable and forking the
        counter would corrupt the n_embed ledger). Returned as an opaque token
        battery25 treats as data.
        """
        return (list(self.entries), [list(v) for v in self.vectors])

    def restore(self, snap) -> None:
        """Restore entries + vectors from a snapshot() result (client/counter/
        embed seam are runtime wiring and are intentionally untouched)."""
        entries, vectors = snap
        self.entries = list(entries)
        self.vectors = [list(v) for v in vectors]

    def reset(self) -> None:
        """Clear all memory (used by the memoryless control, every turn).
        Leaves the client/counter/embed wiring intact."""
        self.entries = []
        self.vectors = []


# ---------------------------------------------------------------------------
# make_backend_g3 -- the standard backend dict (real + stub)
# ---------------------------------------------------------------------------
def _real_backend_g3(rung, counter):
    """Real G3 backend for one rung: a CountingClient over ollama.Client (so the
    chat ceiling still works and every chat is counted), a rung-specific store
    factory, the G3 agent_turn adapter, and battery25 as run_battery.

    Everything runs at qwen3.5:9b with think=False -- the plain CountingClient
    path (mirrors experiments_g2._real_backend; no _ThinkClient needed since
    there is no non-thinking agent tag here). Embeds are counted by the store in
    counter["n_embed"], separate from chat's counter["n_calls"].
    """
    from ollama_client import make_client
    from runner_g25 import battery25

    if rung not in ("R1", "R3"):
        raise KeyError("unknown rung %r; G3 new rungs are R1, R3" % (rung,))

    client = CountingClient_G3(make_client(), counter)

    if rung == "R1":
        def make_store():
            return RawLogStore()
    else:  # R3
        def make_store():
            return VectorStore(client=client, counter=counter)

    def agent_turn(user_msg, store, turn_idx, *, client, model=None,
                   memoryless=False, flavor=None, _respond=None):
        """G3 turn: respond VERBATIM (agent.respond -- no note call, no summary
        call), then append_turn. memoryless=True mirrors G2 semantics: reset the
        store before responding, persist NOTHING. Accepts and IGNORES model= and
        flavor= (passed by run_life), and exposes a _respond injection seam for
        dry tests."""
        if _respond is None:
            from agent import respond as _respond  # lazy, verbatim reuse
        if memoryless:
            store.reset()
        reply = _respond(user_msg, store, client=client, model=MODEL)
        if not memoryless:
            store.append_turn(user_msg, reply)
        return TurnResult(reply=reply, note=None)

    def run_battery(store, *, client, model=None, _respond=None, _judge=None):
        return battery25(store, client=client, agent_model=MODEL,
                         judge_model=JUDGE_MODEL,
                         _respond=_respond, _judge=_judge)

    return {
        "client": client,
        "make_store": make_store,
        "agent_turn": agent_turn,
        "run_battery": run_battery,
        "kind": "real",
    }


class CountingClient_G3:
    """CountingClient over ollama.Client for G3: bumps counter["n_calls"] on every
    .chat/.generate (the chat ceiling), and PASSES .embed through untouched (the
    store counts embeds itself in counter["n_embed"]).

    Mirrors experiments_g2.CountingClient exactly, but is defined here so the
    embed path is explicit: .embed falls through __getattr__ to the inner client,
    NOT wrapped -- so it is never counted against n_calls.
    """

    def __init__(self, inner, counter):
        self._inner = inner
        self._counter = counter

    def chat(self, *args, **kwargs):
        self._counter["n_calls"] += 1
        return self._inner.chat(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self._counter["n_calls"] += 1
        return self._inner.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def make_backend_g3(rung, use_stub, counter):
    """Return a G3 backend dict {client, make_store, agent_turn, run_battery,
    kind} for `rung` ("R1" | "R3"). Consumed VERBATIM by
    experiments_g2.run_life.

    use_stub=True -> delegate to stub_g2.make_stub_backend (deterministic,
    daemon-free code-path validation only; the stub store/agent/battery are the
    G2 fixtures, unrelated to the R1/R3 architectures under test). use_stub=False
    -> the real rung backend that threads qwen3.5:9b and counts embeds
    separately.
    """
    if use_stub:
        from stub_g2 import make_stub_backend
        return make_stub_backend(counter)
    return _real_backend_g3(rung, counter)


# Deferred imports of the frozen types/models used by the real backend, kept at
# module scope so make_backend_g3(use_stub=True) does not require them.
from contract_g2 import TurnResult          # noqa: E402
from contract_g3 import JUDGE_MODEL          # noqa: E402


__all__ = [
    "RawLogStore",
    "VectorStore",
    "make_backend_g3",
    "format_entry",
    "CountingClient_G3",
]
