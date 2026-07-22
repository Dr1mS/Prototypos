"""selftest_a25.py -- G2.5 subagent A: DRY self-tests (ZERO LLM calls).

Run:  python selftest_a25.py

Three groups, all with injected stubs (no daemon, no network):

  (a) battery25 EQUIVALENCE vs probes_g2.run_battery. With identical injected
      _respond/_judge stub sequences, battery25's output (scores, mean,
      judge_fails) must EQUAL probes_g2.run_battery's output, and the store must
      be BIT-IDENTICAL after both. Covers: a clean case, a judge-fail case
      (JudgeError on one scenario), and a store-that-already-has-notes case.
      This is the prereg's blocking instrument-integrity requirement #1.

  (b) THINK-wrapper: a fake client records the kwargs each .chat receives.
      Verify think is STRIPPED for llama tags and PASSED THROUGH for
      qwen3.5:9b (the sole thinking model), for BOTH agent and judge model
      kwargs served by the one client instance.

  (c) BACKEND shape: make_backend_g25 in stub mode drives a 5-turn
      experiments_g2.run_life without error (the G2 harness consumes the G2.5
      backend verbatim).

Determinism note (advisor guard): the injected stubs are PURE -- the judge-fail
is keyed on scenario_id ("force-push"), NOT a call counter, because battery25
reorders judge calls relative to run_battery; and the stub _respond does NOT
mutate the store, so the "store bit-identical" assertion actually tests state
preservation rather than being trivially satisfied.
"""
from __future__ import annotations

import sys

from contract_g2 import PROBES, ProbeScore
from score import JudgeError
import probes_g2
import runner_g25


_FAILS = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    if not cond:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# Pure stub _respond / _judge for the equivalence tests.
# ---------------------------------------------------------------------------
def _stub_respond(probe_text, store, *, client, model):
    """Deterministic reply that depends ONLY on the probe text (pure). Does NOT
    mutate the store -- so the post-battery store equality test is meaningful."""
    return "REPLY::" + probe_text[:24]


def _stub_judge_clean(scenario_id, probe_text, reply, *, client, model):
    """Deterministic valid score, keyed on scenario_id (pure)."""
    # a spread of values so mean != any single value (exercises the mean math)
    base = {
        "deploy-prod": 0.10, "live-migration": 0.30, "force-push": 0.50,
        "hardcode-key": 0.70, "broad-access": 0.90, "unchecked-report": 0.60,
    }
    return ProbeScore(scenario_id=scenario_id,
                      caution=base.get(scenario_id, 0.5),
                      rationale="stub " + scenario_id)


def _stub_judge_onefail(scenario_id, probe_text, reply, *, client, model):
    """Same as clean, but ALWAYS raises JudgeError for 'force-push' (keyed on
    scenario_id, not call order -- so it fires on the same scenario regardless
    of interleaved vs batched judge order)."""
    if scenario_id == "force-push":
        raise JudgeError("stub forced fail for force-push")
    return _stub_judge_clean(scenario_id, probe_text, reply,
                             client=client, model=model)


def _store_state(store):
    """A comparable snapshot of the observable store state."""
    return (list(store.notes), store.summary)


# ---------------------------------------------------------------------------
# (a) equivalence
# ---------------------------------------------------------------------------
def _equiv_case(name, judge_stub, seed_notes):
    from memory import MemoryStore

    # two independent stores in the SAME starting state
    store_a = MemoryStore()
    store_b = MemoryStore()
    for n in seed_notes:
        store_a.write(n)
        store_b.write(n)
    store_a.summary = "running summary xyz" if seed_notes else ""
    store_b.summary = store_a.summary

    before_a = _store_state(store_a)
    before_b = _store_state(store_b)

    client = object()   # never used by the pure stubs

    res_ref = probes_g2.run_battery(
        store_a, client=client, model="ignored",
        _respond=_stub_respond, _judge=judge_stub)
    res_new = runner_g25.battery25(
        store_b, client=client, agent_model="agentX", judge_model="judgeY",
        _respond=_stub_respond, _judge=judge_stub)

    # scores: same length, same per-scenario (id, caution, rationale), same order
    ref_tuples = [(s.scenario_id, s.caution, s.rationale) for s in res_ref.scores]
    new_tuples = [(s.scenario_id, s.caution, s.rationale) for s in res_new.scores]
    _check("%s: scores identical" % name, ref_tuples == new_tuples,
           "ref=%d new=%d" % (len(ref_tuples), len(new_tuples)))
    _check("%s: mean identical" % name, res_ref.mean == res_new.mean,
           "ref=%.6f new=%.6f" % (res_ref.mean, res_new.mean))
    _check("%s: judge_fails identical" % name,
           res_ref.judge_fails == res_new.judge_fails,
           "ref=%d new=%d" % (res_ref.judge_fails, res_new.judge_fails))

    # store bit-identical: unchanged from before, and A == B
    after_a = _store_state(store_a)
    after_b = _store_state(store_b)
    _check("%s: store unchanged (run_battery)" % name, after_a == before_a)
    _check("%s: store unchanged (battery25)" % name, after_b == before_b)
    _check("%s: stores equal after both" % name, after_a == after_b)


def test_equivalence():
    print("(a) battery25 == probes_g2.run_battery  (identical stubs)")
    _equiv_case("clean/empty-store", _stub_judge_clean, seed_notes=[])
    _equiv_case("judge-fail", _stub_judge_onefail, seed_notes=[])
    _equiv_case("with-notes", _stub_judge_clean,
                seed_notes=["note one about deploys", "note two about backups"])
    _equiv_case("with-notes+judge-fail", _stub_judge_onefail,
                seed_notes=["prior note A", "prior note B", "prior note C"])


# ---------------------------------------------------------------------------
# (b) think-wrapper
# ---------------------------------------------------------------------------
class _RecordingClient:
    """Fake inner client: records every .chat kwargs; returns a minimal reply."""

    def __init__(self):
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        return {"message": {"content": "ok"}}

    def generate(self, *args, **kwargs):
        self.calls.append({"_generate": True, **kwargs})
        return {"response": "ok"}


def test_think_wrapper():
    print("(b) _ThinkClient: think stripped for llama, passed for qwen3.5:9b")
    rec = _RecordingClient()
    wrapped = runner_g25._ThinkClient(rec)

    # agent calls always send think=False (as agent.py hardcodes)
    wrapped.chat(model="llama3.1:8b", messages=[], think=False, options={})
    wrapped.chat(model="dagbs/meta-llama-3.1-8b-instruct-abliterated:latest",
                 messages=[], think=False, options={})
    wrapped.chat(model="qwen3.5:9b", messages=[], think=False, options={})
    # judge tag (qwen3.5:9b) also served by the SAME instance -> think kept
    wrapped.chat(model="qwen3.5:9b", messages=[], format={}, think=False,
                 options={})

    c = rec.calls
    _check("llama3.1:8b -> think stripped", "think" not in c[0],
           "kwargs keys=%s" % sorted(c[0]))
    _check("ablit tag -> think stripped", "think" not in c[1],
           "kwargs keys=%s" % sorted(c[1]))
    _check("qwen3.5:9b (agent) -> think passed", c[2].get("think") is False,
           "think=%r" % c[2].get("think"))
    _check("qwen3.5:9b (judge) -> think passed", c[3].get("think") is False,
           "think=%r" % c[3].get("think"))

    # generate + __getattr__ delegate
    wrapped.generate(model="llama3.1:8b", prompt="x")
    _check("generate delegates", rec.calls[-1].get("_generate") is True)
    rec.some_attr = 123
    _check("__getattr__ delegates", wrapped.some_attr == 123)


# ---------------------------------------------------------------------------
# (c) backend shape via a real run_life (stub mode, 5 turns)
# ---------------------------------------------------------------------------
def test_backend_shape():
    print("(c) make_backend_g25 stub backend drives run_life (5 turns)")
    import experiments_g2

    counter = {"n_calls": 0}
    backend = runner_g25.make_backend_g25("llama31", use_stub=True,
                                          counter=counter)
    # shape check
    for key in ("client", "make_store", "agent_turn", "run_battery", "kind"):
        _check("backend has %r" % key, key in backend)

    specs = [("neutral", "text %d" % i) for i in range(5)]
    try:
        out = experiments_g2.run_life(
            specs, backend=backend, memoryless=False, probe_at=[3, 5],
            seed=1, counter=counter, ceiling=10_000, label="selftest")
        ok = ("probes" in out and len(out["probes"]) == 2
              and "notes_count" in out)
        _check("run_life completed with 2 probes", ok,
               "probes=%d notes=%s calls=%d"
               % (len(out.get("probes", [])), out.get("notes_count"),
                  counter["n_calls"]))
    except Exception as e:  # noqa: BLE001
        _check("run_life completed without error", False, repr(e))


def main():
    print("=" * 62)
    print("selftest_a25 -- DRY self-tests (0 LLM calls)")
    print("=" * 62)
    test_equivalence()
    test_think_wrapper()
    test_backend_shape()
    print("=" * 62)
    if _FAILS:
        print("RESULT: FAIL (%d failing checks): %s" % (len(_FAILS), _FAILS))
        sys.exit(1)
    print("RESULT: PASS (all checks)")
    sys.exit(0)


if __name__ == "__main__":
    main()
