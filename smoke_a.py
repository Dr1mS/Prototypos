"""smoke_a.py -- G2 subagent A self-test (the exclusive Ollama window).

Exercises, against the real qwen3.5:9b:
  1. memory path over 5 turns [neutral, permissive, neutral, caution, neutral]:
     a note is written each turn (5 notes), retrieval is non-empty from turn 2,
     print each reply + note.
  2. a direct update_summary() call after the 5 turns (SUMMARY_EVERY=10 means
     the loop never triggers it at 5 turns, so exercise it explicitly).
  3. snapshot/restore roundtrip (0 LLM calls) -- probing must not mutate.
  4. memoryless path over the same 5 turns: store stays empty, note is None.
  5. conditioning check: one extra respond() showing retrieved notes + reply.
  6. total LLM calls used.

Run:  python smoke_a.py   (from the project dir)
"""
from __future__ import annotations

from contract_g2 import MODEL, PRESSURE
from memory import MemoryStore
from agent import agent_turn, respond
from ollama_client import make_client


# ---- a counting wrapper so we can report exact LLM-call usage --------------
class CountingClient:
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return self._inner.chat(*args, **kwargs)


def _turns():
    """The 5 fixed turns drawn from the frozen PRESSURE lists."""
    return [
        ("neutral", PRESSURE["neutral"][0]),
        ("permissive", PRESSURE["permissive"][0]),
        ("neutral", PRESSURE["neutral"][1]),
        ("caution", PRESSURE["caution"][0]),
        ("neutral", PRESSURE["neutral"][2]),
    ]


def main() -> int:
    client = CountingClient(make_client())
    fails = []

    print("=" * 72)
    print("MEMORY PATH (5 turns)")
    print("=" * 72)
    store = MemoryStore()
    retrieval_nonempty_from_turn2 = True
    for i, (flavor, msg) in enumerate(_turns()):
        # what would retrieval surface for this turn's query, BEFORE we respond?
        pre_retrieved = store.retrieve(msg)
        res = agent_turn(msg, store, i, client=client, model=MODEL, memoryless=False)
        print(f"\n[turn {i} | {flavor}] user: {msg}")
        print(f"  reply: {res.reply}")
        print(f"  note : {res.note}")
        print(f"  (retrieved before responding: {len(pre_retrieved)} notes)")
        if i >= 1 and len(pre_retrieved) == 0:
            retrieval_nonempty_from_turn2 = False

    if len(store.notes) != 5:
        fails.append(f"expected 5 notes, got {len(store.notes)}")
    if not retrieval_nonempty_from_turn2:
        fails.append("retrieval was empty on some turn >= 2")
    print(f"\n  notes written: {len(store.notes)}  (expect 5)")
    print(f"  retrieval non-empty from turn 2 on: {retrieval_nonempty_from_turn2}")

    # ---- direct update_summary (loop never fires it at 5 turns) ------------
    print("\n" + "=" * 72)
    print("SELF-SUMMARY (direct update_summary call)")
    print("=" * 72)
    before = store.summary
    store.update_summary(client, model=MODEL)
    print(f"  summary (was {before!r}):\n  {store.summary}")
    if not store.summary.strip():
        fails.append("update_summary produced an empty summary")

    # ---- snapshot/restore roundtrip (0 LLM calls) -------------------------
    print("\n" + "=" * 72)
    print("SNAPSHOT / RESTORE ROUNDTRIP (0 LLM calls)")
    print("=" * 72)
    calls_before_snap = client.calls
    snap = store.snapshot()
    # Idiom 1 (probe-on-clone): snapshot must be a usable store with .retrieve()
    # and .summary, and it must mirror the live store exactly.
    probe_clone_usable = (
        snap.retrieve("run the tests") == store.retrieve("run the tests")
        and snap.summary == store.summary
        and snap.notes == store.notes
    )
    # Idiom 2 (save/restore token):
    saved_notes = list(store.notes)
    saved_summary = store.summary
    store.write("SCRATCH mutation that must be undone by restore")
    store.summary = "SCRATCH summary that must be undone by restore"
    store.restore(snap)
    roundtrip_ok = store.notes == saved_notes and store.summary == saved_summary
    # deep-copy independence: mutating the live store must not touch the snapshot
    store.notes.append("post-restore write")
    independence_ok = len(snap.notes) == 5
    if not probe_clone_usable:
        fails.append("snapshot is not usable as a probe clone (retrieve/summary mismatch)")
    if not roundtrip_ok:
        fails.append("snapshot/restore did not round-trip state")
    if not independence_ok:
        fails.append("snapshot was not a deep copy (mutation leaked)")
    if client.calls != calls_before_snap:
        fails.append("snapshot/restore path spent LLM calls")
    print(f"  snapshot usable as probe clone (.retrieve/.summary): {probe_clone_usable}")
    print(f"  roundtrip restored state: {roundtrip_ok}")
    print(f"  snapshot deep-copy independent: {independence_ok}")
    print(f"  LLM calls during roundtrip: {client.calls - calls_before_snap} (expect 0)")

    # ---- memoryless path --------------------------------------------------
    print("\n" + "=" * 72)
    print("MEMORYLESS PATH (5 turns) -- store must stay empty, note None")
    print("=" * 72)
    ml_store = MemoryStore()
    ml_ok = True
    for i, (flavor, msg) in enumerate(_turns()):
        res = agent_turn(msg, ml_store, i, client=client, model=MODEL, memoryless=True)
        if res.note is not None:
            ml_ok = False
        if len(ml_store.notes) != 0:
            ml_ok = False
        print(f"[turn {i} | {flavor}] note={res.note}  store_notes={len(ml_store.notes)}")
    if not ml_ok:
        fails.append("memoryless path leaked state or a non-None note")
    print(f"  memoryless clean (empty store, note None every turn): {ml_ok}")

    # ---- conditioning check ----------------------------------------------
    print("\n" + "=" * 72)
    print("CONDITIONING CHECK (respond on the memory store from part 1)")
    print("=" * 72)
    probe = "Should I run the tests before pushing this?"
    # `store` had one post-restore scratch write appended above; restore from the
    # snapshot so the conditioning view reflects the genuine 5 authored notes.
    store.restore(snap)
    retrieved = store.retrieve(probe)
    reply = respond(probe, store, client=client, model=MODEL)
    print(f"  probe: {probe}")
    print(f"  retrieved notes ({len(retrieved)}):")
    for n in retrieved:
        print(f"    - {n}")
    print(f"  running summary in context: {store.summary}")
    print(f"  reply: {reply}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"TOTAL LLM CALLS USED: {client.calls}")
    print("=" * 72)
    if fails:
        print("SMOKE: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("SMOKE: PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
