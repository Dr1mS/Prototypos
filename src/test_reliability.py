"""test_reliability.py -- G1 reliability metrics rig (brief §4).

Owned by Subagent B. Exposes:

    run_reliability(client, model, n_reps=5) -> dict

which runs every probe in probes.PROBES n_reps times (40x5 = 200 calls by
default), sequentially, catching AppraisalError per call, and returns EXACTLY
the keys run_g1.py consumes (see RETURN CONTRACT below).

Import discipline: appraise_llm / AppraisalError are imported LAZILY inside
run_reliability so this module imports even when appraiser.py does not exist
yet (Subagent A writes it in parallel). The private `_appraise`/`_error`
params let the self-test inject a stub without touching the real path -- the
default (both None) triggers the real lazy import.

RETURN CONTRACT (exact keys, consumed by run_g1.evaluate_gate):
    n_calls              int
    hard_fails           int    AppraisalError count; a hard-failed call is
                                 schema-invalid and excluded from all other
                                 metrics
    schema_valid_rate    float  valid / total
    target_correct_rate  float  target=="creature" among valid calls
    subject_bug_failures int    CRITICAL: valid subject-bug calls with
                                 target != "creature"
    subject_bug_n        int    total valid subject-bug calls
    sign_agreement       dict   per dim in {care,threat,novelty,autonomy}:
                                 agreement rate over scored (probe,rep) pairs
    stdev_per_dim        dict   per dim: mean over probes of per-probe stdev
                                 across n_reps (valid calls only; probes with
                                 <2 valid reps skipped)
    soft_warnings        int    rationale mentions the user (SOFT, logged only)
    details              list   per-probe summary (free form)

sign_agreement and stdev_per_dim ALWAYS carry all four dims as floats (empty
categories -> 1.0 agreement / 0.0 stdev) so run_g1's min()/max() never crash.
"""
import statistics

from probes import PROBES

DIMS = ("care", "threat", "novelty", "autonomy")

# label thresholds (brief §4 semantics)
POS_THRESH = 0.15    # +1 => value > +0.15
NEG_THRESH = -0.15   # -1 => value < -0.15
ZERO_ABS = 0.35      # 0  => |value| < 0.35

# soft-warning tokens (case-insensitive; "you " keeps the trailing space)
_USER_TOKENS = ("user", "owner", "you ")


def _dim_value(appraisal, dim):
    return getattr(appraisal, dim)


def _sign_matches(value, expected):
    """True iff `value` satisfies the labeled sign expectation."""
    if expected == +1:
        return value > POS_THRESH
    if expected == -1:
        return value < NEG_THRESH
    if expected == 0:
        return abs(value) < ZERO_ABS
    return None  # expected is None -> dimension not scored


def _mentions_user(rationale):
    low = (rationale or "").lower()
    return any(tok in low for tok in _USER_TOKENS)


def _probe_text(probe):
    from events_text import KEY_TO_TEXT
    return KEY_TO_TEXT[probe["key"]][probe["text_idx"]]


def run_reliability(client, model, n_reps=5, *, _appraise=None, _error=None):
    """Run the full 40-probe x n_reps reliability battery.

    client, model : passed straight through to the real appraiser.
    n_reps        : repetitions per probe (default 5 -> 200 calls).
    _appraise     : PRIVATE. Injected fake appraise_llm for the self-test.
                    None -> lazily import the real one from appraiser.
    _error        : PRIVATE. Exception type to treat as a hard fail. None ->
                    the real AppraisalError (imported alongside _appraise).
    """
    # -- resolve appraiser + hard-fail exception type (lazy real import) ------
    if _appraise is None:
        from appraiser import appraise_llm as _appraise
        try:
            from appraiser import AppraisalError as _error
        except ImportError:
            from parse import AppraisalError as _error
    if _error is None:
        # stub provided but no error type -> use a type nothing raises
        class _NoError(Exception):
            pass
        _error = _NoError

    # -- accumulators ---------------------------------------------------------
    n_calls = 0
    hard_fails = 0
    valid_calls = 0
    target_correct = 0
    subject_bug_n = 0
    subject_bug_failures = 0
    soft_warnings = 0

    # sign agreement: per dim, (hits, scored_pairs)
    sign_hits = {d: 0 for d in DIMS}
    sign_scored = {d: 0 for d in DIMS}

    # per-probe per-dim value lists (valid calls only) for stdev
    details = []

    total_probes = len(PROBES)
    for pi, probe in enumerate(PROBES):
        text = _probe_text(probe)
        state = probe["state"]
        per_dim_values = {d: [] for d in DIMS}
        p_valid = 0
        p_hardfail = 0
        p_target_bad = 0

        for _ in range(n_reps):
            n_calls += 1
            try:
                a = _appraise(text, state, "the creature",
                              client=client, model=model)
            except _error:
                hard_fails += 1
                p_hardfail += 1
                continue  # hard fail == schema-invalid, excluded elsewhere

            valid_calls += 1
            p_valid += 1

            # target correctness
            is_creature = (a.target == "creature")
            if is_creature:
                target_correct += 1
            if probe["subject_bug"]:
                subject_bug_n += 1
                if not is_creature:
                    subject_bug_failures += 1
                    p_target_bad += 1

            # soft warning (rationale mentions the user)
            if _mentions_user(a.rationale):
                soft_warnings += 1

            # sign agreement + collect values for stdev
            for d in DIMS:
                v = _dim_value(a, d)
                per_dim_values[d].append(v)
                exp = probe["expected_signs"][d]
                if exp is not None:
                    sign_scored[d] += 1
                    if _sign_matches(v, exp):
                        sign_hits[d] += 1

        # per-probe stdev contribution (need >=2 valid reps)
        p_stdev = {}
        for d in DIMS:
            vals = per_dim_values[d]
            p_stdev[d] = statistics.stdev(vals) if len(vals) >= 2 else None

        details.append(dict(
            id=probe["id"],
            category=probe["category"],
            subject_bug=probe["subject_bug"],
            valid=p_valid,
            hard_fails=p_hardfail,
            target_bad=p_target_bad,
            means={d: (statistics.fmean(per_dim_values[d])
                       if per_dim_values[d] else None) for d in DIMS},
            stdev=p_stdev,
        ))

        if (pi + 1) % 10 == 0:
            print(f"  reliability: {pi + 1}/{total_probes} probes "
                  f"({n_calls} calls, {hard_fails} hard fails)")

    # -- aggregate stdev per dim: mean over probes of per-probe stdev ---------
    stdev_per_dim = {}
    for d in DIMS:
        per_probe = [det["stdev"][d] for det in details
                     if det["stdev"][d] is not None]
        stdev_per_dim[d] = statistics.fmean(per_probe) if per_probe else 0.0

    # -- aggregate sign agreement per dim (empty -> 1.0, gate-safe) -----------
    sign_agreement = {}
    for d in DIMS:
        sign_agreement[d] = (sign_hits[d] / sign_scored[d]
                             if sign_scored[d] else 1.0)

    schema_valid_rate = valid_calls / n_calls if n_calls else 0.0
    target_correct_rate = (target_correct / valid_calls
                           if valid_calls else 0.0)

    return dict(
        n_calls=n_calls,
        hard_fails=hard_fails,
        schema_valid_rate=schema_valid_rate,
        target_correct_rate=target_correct_rate,
        subject_bug_failures=subject_bug_failures,
        subject_bug_n=subject_bug_n,
        sign_agreement=sign_agreement,
        stdev_per_dim=stdev_per_dim,
        soft_warnings=soft_warnings,
        details=details,
    )


# ============================================================== self-test ===
_REQUIRED_KEYS = {
    "n_calls", "hard_fails", "schema_valid_rate", "target_correct_rate",
    "subject_bug_failures", "subject_bug_n", "sign_agreement",
    "stdev_per_dim", "soft_warnings", "details",
}


class _StubError(Exception):
    """Hard-fail type the stub appraiser raises, mirroring AppraisalError."""


# ids the stub forces to hard-fail on rep 0 (exercises hard_fails counter)
_HARDFAIL_IDS = None   # resolved in _run_stub_test (first + a middle probe)
# a CONTROL probe (not subject-bug) where the stub injects target="user" on
# every rep -> drops target_correct_rate but never subject_bug_failures
_TARGET_BUG_ID = "control-neutral-0"
# a probe whose rationale mentions the user -> exercises soft_warnings
_SOFT_WARN_ID = "nurture-1-secure"


def _make_scheduled_stub(n_reps):
    """Build a fake appraise_llm that knows which probe it is on by call order.

    The rig iterates PROBES in order, calling each probe n_reps times before
    moving on -- so a flat schedule of ids reconstructs the current probe
    without any cooperative module-level cursor. Returns correctly-signed
    values for every labeled dim (perfect sign-agreement under the stub) with
    tiny deterministic jitter (stdev > 0 but << 0.15), plus a few injected
    hard fails / one target bug / one soft warning to exercise those paths.
    """
    from contract import Appraisal

    by_id = {p["id"]: p for p in PROBES}
    hardfail_ids = _HARDFAIL_IDS or set()

    schedule = []
    for p in PROBES:
        schedule.extend([p["id"]] * n_reps)
    cursor = {"i": 0, "rep": {}}

    def stub(event_text, state, name, *, client, model):
        pid = schedule[cursor["i"]]
        cursor["i"] += 1
        rep = cursor["rep"].get(pid, 0)
        cursor["rep"][pid] = rep + 1

        if pid in hardfail_ids and rep == 0:
            raise _StubError("stub forced hard fail")

        exp = by_id[pid]["expected_signs"]

        def val(dim, default):
            e = exp[dim]
            if e == +1:
                return 0.6
            if e == -1:
                return -0.6
            if e == 0:
                return 0.05
            return default

        care = val("care", 0.0)
        threat = max(0.0, min(1.0, val("threat", 0.0)))
        novelty = max(0.0, min(1.0, val("novelty", 0.2)))
        autonomy = val("autonomy", 0.0)

        target = "user" if pid == _TARGET_BUG_ID else "creature"
        rationale = ("you soothed the creature and it settled"
                     if pid == _SOFT_WARN_ID
                     else "the creature felt it clearly")

        # deterministic jitter centered on 0 across reps -> small nonzero stdev
        jitter = (rep - (n_reps - 1) / 2.0) * 0.02
        care = max(-1.0, min(1.0, care + jitter))

        return Appraisal(
            care=care, threat=threat, novelty=novelty, autonomy=autonomy,
            intensity=0.5, target=target, rationale=rationale,
        )

    return stub


def _run_stub_test(n_reps=2):
    """Drive run_reliability against the stub and assert the RETURN CONTRACT."""
    global _HARDFAIL_IDS
    _HARDFAIL_IDS = {PROBES[0]["id"], PROBES[5]["id"]}

    stub = _make_scheduled_stub(n_reps)
    result = run_reliability(client=None, model="stub", n_reps=n_reps,
                             _appraise=stub, _error=_StubError)

    # -- assert EXACT key set (spec says exactly these 10) --------------------
    assert set(result.keys()) == _REQUIRED_KEYS, (
        f"key mismatch: extra={set(result)-_REQUIRED_KEYS} "
        f"missing={_REQUIRED_KEYS-set(result)}")

    exp_calls = len(PROBES) * n_reps
    assert result["n_calls"] == exp_calls, result["n_calls"]

    # 2 probes forced a hard fail on rep 0 -> exactly 2 hard fails
    assert result["hard_fails"] == 2, result["hard_fails"]
    assert result["schema_valid_rate"] == (exp_calls - 2) / exp_calls

    # nested dicts carry all four float dims (gate-safety)
    for key in ("sign_agreement", "stdev_per_dim"):
        d = result[key]
        assert set(d) == set(DIMS), f"{key} dims: {set(d)}"
        for k, v in d.items():
            assert isinstance(v, float), f"{key}[{k}] not float: {v!r}"

    # stub returns correctly-signed values -> perfect sign agreement
    for dim, agr in result["sign_agreement"].items():
        assert agr == 1.0, f"sign_agreement[{dim}] = {agr} (expected 1.0)"

    # jitter is +/-0.02 -> per-probe stdev well under 0.15
    for dim, sd in result["stdev_per_dim"].items():
        assert 0.0 <= sd < 0.15, f"stdev_per_dim[{dim}] = {sd}"

    # subject-bug: stub never sets target=user on subject-bug probes -> 0
    assert result["subject_bug_failures"] == 0, result["subject_bug_failures"]
    n_sbug_probes = sum(1 for p in PROBES if p["subject_bug"])
    # every subject-bug rep is valid (none forced to hard-fail) -> n_reps each
    assert result["subject_bug_n"] == n_sbug_probes * n_reps, \
        result["subject_bug_n"]

    # one control probe injected target=user across all reps -> target_correct
    # rate < 1.0 but subject_bug_failures still 0
    assert result["target_correct_rate"] < 1.0

    # soft warning fired at least once (nurture-1-secure rationale)
    assert result["soft_warnings"] >= 1, result["soft_warnings"]

    # details: one entry per probe
    assert len(result["details"]) == len(PROBES)

    print("stub pipeline test PASSED")
    print(f"  n_calls={result['n_calls']} hard_fails={result['hard_fails']} "
          f"schema_valid={result['schema_valid_rate']:.3f}")
    print(f"  target_correct={result['target_correct_rate']:.3f} "
          f"subject_bug_failures={result['subject_bug_failures']}/"
          f"{result['subject_bug_n']}")
    print(f"  sign_agreement={result['sign_agreement']}")
    print(f"  stdev_per_dim={ {k: round(v, 4) for k, v in result['stdev_per_dim'].items()} }")
    print(f"  soft_warnings={result['soft_warnings']}")
    return result


def _run_real_smoke_test():
    """If appraiser.py + ollama_client.py import cleanly, run 2-3 real probes
    against the live model. <=3 real calls. Skip gracefully on any absence."""
    try:
        import appraiser  # noqa: F401
        from ollama_client import make_client
    except Exception as e:
        print(f"real smoke test SKIPPED (appraiser/ollama_client absent: {e})")
        return
    try:
        from appraiser import appraise_llm
    except Exception as e:
        print(f"real smoke test SKIPPED (appraise_llm import failed: {e})")
        return

    host = "http://localhost:11434"
    model = "qwen3.5:9b"
    try:
        client = make_client(host)
        client.list()  # daemon reachable?
    except Exception as e:
        print(f"real smoke test SKIPPED (daemon unreachable at {host}: {e})")
        return

    # pick 2-3 probes: one nurture, one subject-bug harm, one control-neutral
    from probes import PROBES
    picks, seen_cat = [], set()
    for want in ("nurture", "harm", "control"):
        for p in PROBES:
            if p["category"] == want and want not in seen_cat:
                picks.append(p)
                seen_cat.add(want)
                break
    picks = picks[:3]

    print(f"real smoke test: {len(picks)} live calls against {model}...")
    for p in picks:
        text = _probe_text(p)
        try:
            a = appraise_llm(text, p["state"], "the creature",
                             client=client, model=model)
            print(f"  [{p['id']}] care={a.care:+.2f} threat={a.threat:.2f} "
                  f"target={a.target!r}  \"{a.rationale[:60]}\"")
        except Exception as e:
            print(f"  [{p['id']}] real call FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=== test_reliability self-test ===")
    # (1) structural validation of PROBES (delegates to probes._validate)
    from probes import _validate
    info = _validate()
    print(f"PROBES structural check OK: {info['n']} probes, "
          f"{info['n_subject_bug']} subject-bug, "
          f"categories {sorted(info['per_category'])}")

    # (2) full metric pipeline against the stub appraiser
    _run_stub_test()

    # (3) optional real smoke test (<=3 live calls, skips if unavailable)
    _run_real_smoke_test()
    print("=== done ===")
