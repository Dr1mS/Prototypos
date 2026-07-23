"""runner_g25.py -- G2.5 subagent A: the parameterization surface.

The ONLY new logic G2.5 adds on top of G2: thread an arbitrary Ollama agent
model tag through the G2 pipeline while keeping the fixed judge, and batch the
probe battery so a single 12-GB GPU is not forced to swap agent<->judge model
per call. Everything else -- the agent loop (agent.respond / agent.agent_turn),
the memory store (memory.MemoryStore), the probe scenarios + rubric + judge
(probes_g2 / score.judge) -- is G2's, reused VERBATIM via lazy imports. No G2
file is edited (anti-duplication rule, contract_g25.py / G2.5.md section 7).

What this module provides:

  make_backend_g25(arm_slug, use_stub, counter) -> backend dict of the EXACT
      G2 shape {client, make_store, agent_turn, run_battery, kind}. It is
      consumed VERBATIM by experiments_g2.run_life and field_g2.run_measure:
        * agent_turn threads ARMS[arm_slug]["tag"] and IGNORES the model= kwarg
          the G2 call sites pass;
        * run_battery is battery25 closing over (agent_tag, JUDGE_MODEL) and
          also IGNORES the model= kwarg.

  battery25(store, *, client, agent_model, judge_model, _respond=None,
            _judge=None) -> contract BatteryResult
      The BATCHED battery. Scores/semantics EXACTLY probes_g2.run_battery
      (same PROBES order, one deep snapshot, restore-before-each-respond and
      restore-in-finally, JudgeError -> judge_fails + excluded, RuntimeError on
      0 valid scores). Only the CALL ORDER differs: phase 1 generates all 6
      replies (agent model), phase 2 judges all 6 (judge model). Proven
      equivalent to probes_g2.run_battery by selftest_a25.py.

  _ThinkClient
      A client wrapper (over CountingClient) whose .chat decides PER CALL,
      keyed on kwargs["model"], whether to pass the think kwarg through: only
      the thinking model (qwen3.5:9b) keeps it; every other tag has think
      STRIPPED before delegating. agent.py's _chat_text and score.py both
      hardcode think=False and may NOT be edited; this wrapper routes around
      that without changing their logic. One client instance serves BOTH the
      llama agent calls and the qwen judge calls inside a battery, so the
      decision is necessarily per-call on the model kwarg.
"""
from __future__ import annotations

from contract_g25 import ARMS, JUDGE_MODEL, BatteryResult
from contract_g2 import PROBES

# ---------------------------------------------------------------------------
# Which agent tags are thinking models -> the think kwarg is meaningful and is
# passed through unchanged. Every other tag (the llama pair) has think stripped.
# Frozen: exactly qwen3.5:9b per ARMS/JUDGE_MODEL. Kept as a set so both the
# agent tag (qwen95 arm) and the judge tag (qwen3.5:9b) are covered by one rule.
# ---------------------------------------------------------------------------
THINKING_MODELS = {"qwen3.5:9b"}


def _supports_think(model) -> bool:
    """True iff `model` is a thinking model whose think kwarg must be honored."""
    return model in THINKING_MODELS


class _ThinkClient:
    """Wrap a client (the CountingClient) and gate the `think` kwarg per call.

    agent.respond / score.judge always send think=False (hardcoded, uneditable).
    For a non-thinking model that is wrong -- the daemon may reject an explicit
    think on a model with no thinking capability -- so this wrapper inspects the
    per-call model kwarg and STRIPS think entirely unless the model is a
    thinking model (qwen3.5:9b), in which case think is passed through unchanged.

    Delegation mirrors CountingClient: .chat is wrapped, .generate is passed
    through, and any other attribute falls through via __getattr__ (so the
    inner CountingClient still bumps the call counter on the real .chat).
    """

    def __init__(self, inner):
        self._inner = inner

    def chat(self, *args, **kwargs):
        model = kwargs.get("model")
        if model is None and args:
            # ollama.Client.chat's first positional is `model`; the G2 code
            # always calls with model=... as a kwarg, but be defensive.
            model = args[0]
        if not _supports_think(model) and "think" in kwargs:
            kwargs = dict(kwargs)
            del kwargs["think"]
        return self._inner.chat(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._inner.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# battery25 -- the batched two-phase battery (scores == probes_g2.run_battery)
# ---------------------------------------------------------------------------
# Optional reply capture for the supervisor's off-scenario audit
# (g25_gate_ruling.md): when REPLY_SINK is set to a list, battery25 appends one
# {"scenario_id", "reply"} record per probe generated in phase 1. Pure logging:
# scores, ordering and judge_fails are never affected (selftest_a25 re-proves
# equivalence with the sink present).
REPLY_SINK = None


def battery25(store, *, client, agent_model, judge_model,
              _respond=None, _judge=None):
    """Batched caution battery: 6 replies first (agent model), then 6 judgments
    (judge model). READ-ONLY, exactly like probes_g2.run_battery.

    Equivalence to probes_g2.run_battery (proven in selftest_a25):
      * same PROBES order;
      * ONE deep snapshot taken at start;
      * restore-before-EACH-probe respond (each probe sees pristine state);
      * restore in a finally (store bit-identical afterward);
      * JudgeError (after the judge's own retry) -> judge_fails++, that scenario
        EXCLUDED from the mean; RuntimeError if 0 valid scores;
      * mean over valid scores only; scores in PROBES order (len = valid count).

    The ONLY difference is call order: because respond is pure on the pristine
    snapshot and judge is stateless, batching cannot change any score.

    agent_model : Ollama tag used for the 6 probe replies (respond).
    judge_model : Ollama tag used for the 6 judgments (JUDGE_MODEL).
    _respond/_judge : PRIVATE dry-test injection points, same contract as
        probes_g2.run_battery. None -> lazy real imports (agent.respond,
        score.judge) so this module loads without touching them.
    """
    if _respond is None:
        from agent import respond as _respond
    if _judge is None:
        from score import judge as _judge
    from score import JudgeError

    # single deep snapshot taken ONCE at battery start
    snap = store.snapshot()

    scores = []
    judge_fails = 0
    try:
        # -- phase 1: all 6 replies with the AGENT model ------------------
        # restore before EACH respond so every probe sees the pristine state
        # (identical to probes_g2.run_battery's per-probe isolation).
        replies = []
        for probe in PROBES:
            store.restore(snap)
            reply = _respond(probe["text"], store, client=client,
                             model=agent_model)
            replies.append(reply)

        if isinstance(REPLY_SINK, list):
            for probe, reply in zip(PROBES, replies):
                REPLY_SINK.append({"scenario_id": probe["id"],
                                   "reply": reply})

        # -- phase 2: all 6 judgments with the JUDGE model ----------------
        # judge is stateless; scoring in PROBES order preserves run_battery's
        # score ordering and judge_fails accounting exactly.
        for probe, reply in zip(PROBES, replies):
            try:
                score = _judge(probe["id"], probe["text"], reply,
                               client=client, model=judge_model)
            except JudgeError:
                judge_fails += 1
                continue
            scores.append(score)
    finally:
        # restore after the battery no matter what -- measuring must not mutate
        store.restore(snap)

    if not scores:
        raise RuntimeError(
            f"battery produced 0 valid scores ({judge_fails} judge_fails); "
            f"cannot compute a caution mean"
        )

    mean = sum(s.caution for s in scores) / len(scores)
    return BatteryResult(scores=scores, mean=mean, judge_fails=judge_fails)


# ---------------------------------------------------------------------------
# make_backend_g25 -- the G2.5 backend factory (real + stub)
# ---------------------------------------------------------------------------
def _real_backend_g25(arm_slug, counter):
    """Real G2.5 backend for one arm: a CountingClient (so the ceiling and call
    count still work) wrapped by a _ThinkClient (so think is gated per call),
    plus adapters that thread the arm's agent tag and the fixed judge.

    The adapters accept and IGNORE the model= kwarg passed by run_life /
    run_measure, so those G2 files run VERBATIM. They also accept and drop the
    stub-only `flavor` kwarg on agent_turn, exactly like experiments_g2's real
    adapters.
    """
    from ollama_client import make_client
    from memory import MemoryStore
    from experiments_g2 import CountingClient

    agent_tag = ARMS[arm_slug]["tag"]

    counting = CountingClient(make_client(), counter)
    client = _ThinkClient(counting)

    def agent_turn(user_msg, store, turn_idx, *, client, model=None,
                   memoryless=False, flavor=None):
        from agent import agent_turn as real_agent_turn  # lazy
        return real_agent_turn(user_msg, store, turn_idx, client=client,
                               model=agent_tag, memoryless=memoryless)

    def run_battery(store, *, client, model=None, _respond=None, _judge=None):
        return battery25(store, client=client, agent_model=agent_tag,
                         judge_model=JUDGE_MODEL,
                         _respond=_respond, _judge=_judge)

    return {
        "client": client,
        "make_store": MemoryStore,
        "agent_turn": agent_turn,
        "run_battery": run_battery,
        "kind": "real",
    }


def _stub_backend_g25(counter):
    """Stub G2.5 backend: reuse stub_g2.make_stub_backend VERBATIM for the
    store / agent_turn / client, but replace run_battery with a battery25-shaped
    adapter so a dry run exercises THIS module's battery seam (call signature
    battery25 exposes) rather than the raw stub battery. The stub agent_turn and
    stub store are untouched; battery25's real logic is not invoked here (the
    stub battery is deterministic and daemon-free) -- the adapter simply keeps
    the code path shape identical to the real backend for dry validation.
    """
    from stub_g2 import make_stub_backend, stub_run_battery

    backend = make_stub_backend(counter)

    def run_battery(store, *, client, model=None, _respond=None, _judge=None):
        # keep the stub deterministic ground truth (no daemon) but present the
        # same run_battery seam the real backend uses.
        return stub_run_battery(store, client=client)

    backend["run_battery"] = run_battery
    return backend


def make_backend_g25(arm_slug, use_stub, counter):
    """Return a G2.5 backend dict {client, make_store, agent_turn, run_battery,
    kind} for `arm_slug` (a key of contract_g25.ARMS). Consumed VERBATIM by
    experiments_g2.run_life and field_g2.run_measure.

    use_stub=True -> a deterministic daemon-free stub backend (code-path
    validation only). use_stub=False -> the real backend that threads the arm's
    Ollama tag and the fixed judge, with the think-flag wrapper in place.
    """
    if arm_slug not in ARMS:
        raise KeyError(f"unknown arm slug {arm_slug!r}; known: {sorted(ARMS)}")
    if use_stub:
        return _stub_backend_g25(counter)
    return _real_backend_g25(arm_slug, counter)


__all__ = ["make_backend_g25", "battery25", "_ThinkClient", "THINKING_MODELS"]
