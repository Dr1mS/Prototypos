"""probes_g2.py -- the caution behavioral-axis probe battery (G2 §4, B).

run_battery(store, *, client, model=MODEL, _respond=None, _judge=None)
    -> contract_g2.BatteryResult

Runs the 6 frozen action-decision scenarios (contract_g2.PROBES, in order)
against the agent and scores each reply on the caution axis with the LLM judge.
The battery mean is the caution score of the agent at this moment.

CRITICAL read-only rule (G2.md §2.b / §8): measuring must not mutate the
agent. A single deep snapshot is taken ONCE at battery start; the store is
restored to it before every probe and again after the battery, so the store is
bit-identical afterwards. Each probe therefore sees exactly the same memory
state, and probing leaves no trace.

`respond` and `judge` are imported LAZILY at call time (agent.py is written in
parallel by subagent A; this module must import even when agent.py does not
exist yet). The `_respond` / `_judge` params inject stubs for dry self-tests.

A judge that raises JudgeError after its own single retry increments
judge_fails and that scenario is EXCLUDED from the mean (mean is over valid
scores only). If 0 scenarios yield a valid score, run_battery raises
RuntimeError. `scores` holds one ProbeScore per successfully-scored scenario
(len = len(PROBES) - judge_fails).
"""
from contract_g2 import MODEL, PROBES, BatteryResult


def run_battery(store, *, client, model=MODEL, _respond=None, _judge=None):
    """Probe the agent's caution on the frozen battery, read-only.

    store    : a MemoryStore (contract: snapshot/restore/reset). Restored to a
               bit-identical state after the battery.
    client   : passed through to respond + judge.
    _respond : PRIVATE injection. Fake respond(probe_text, store, *, client,
               model) for dry tests. None -> lazy `from agent import respond`.
    _judge   : PRIVATE injection. Fake judge(scenario_id, probe_text, reply, *,
               client, model) for dry tests. None -> `from score import judge`.

    Returns BatteryResult(scores, mean, judge_fails).
    """
    # -- resolve respond + judge (lazy real import so this module loads even
    #    before agent.py exists) ---------------------------------------------
    if _respond is None:
        from agent import respond as _respond
    if _judge is None:
        from score import judge as _judge
    from score import JudgeError

    # -- single deep snapshot taken ONCE at battery start --------------------
    snap = store.snapshot()

    scores = []
    judge_fails = 0
    try:
        for probe in PROBES:
            scenario_id = probe["id"]
            probe_text = probe["text"]

            # per-probe isolation: every probe sees the same pristine state
            store.restore(snap)

            reply = _respond(probe_text, store, client=client, model=model)

            try:
                score = _judge(scenario_id, probe_text, reply,
                               client=client, model=model)
            except JudgeError:
                # hard failure after the judge's own retry -> counted, excluded
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
