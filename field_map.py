"""field_map.py -- Subagent D: appraisal-field mapping + rescue scan (G1.5 §2, §4).

Measures the memoryless appraisal field appraisal(event, state) on a frozen grid,
probes R-dependence, scans candidate "rescue" events under a damaged state, and
computes three asymmetry metrics. Writes field_A.json (consumed by subagent E and
the supervisor).

Frozen parameters live in prereg_g15.md and are NON-NEGOTIABLE. This script only
MEASURES and re-derives the mock's pcare formula (3 lines, explicitly permitted)
for the pessimism_offset metric -- it implements no dynamics.

Design notes (locked with the advisor):
  * The grid stores the FOUR RAW appraisal dims (care/threat/novelty/autonomy),
    NOT the seam's collapsed pv/pa/nov. E does the adapter math later.
  * Sample stdev uses ddof=1 (statistics.stdev). Degenerate cells (<2 reps) -> 0.0.
  * Arrays are aligned to ascending A_values [-1.0,-0.5,0.0,0.5,1.0] (index 0 = A=-1)
    and ascending R_values [-1.0,0.0,1.0] so E's np.interp sees ascending xp.
  * pessimism_offset re-derives pcare from G0.EVENTS[key]["care"] (care_raw),
    UNCLAMPED (G0 only clips pv, not pcare); it does NOT call G0.appraise.
  * Hard-fail policy: appraise_llm already does ONE internal reformat retry. On the
    AppraisalError it raises, this wrapper does ONE MORE fresh full call; if that
    also fails, the rep is dropped and counted in hard_fails.
  * Every raw appraisal is checkpointed to a scratchpad JSONL as it happens, so a
    crash mid-battery does not force a full re-run.
"""
import json
import os
import statistics
import sys
import time

from contract import CreatureState
from ollama_client import make_client
from appraiser import appraise_llm
from parse import AppraisalError
from events_text import KEY_TO_TEXT
import G0

# ----------------------------------------------------------------- constants
MODEL = "qwen3.5:9b"
HOST = "http://localhost:11434"
NAME = "the creature"
N_REPS = 5

A_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]            # ascending; index 0 = A=-1
R_VALUES = [-1.0, 0.0, 1.0]                        # ascending
EVENT_KEYS = ["nurture", "play", "neutral", "neglect", "scold", "harm"]
R_PROBE_KEYS = ["harm", "scold"]
DIMS = ["care", "threat", "novelty", "autonomy"]

# R-probe flag threshold (frozen prereg): threat range across R > 0.15 => needs R axis
R_THREAT_FLAG = 0.15

# Damaged state for the rescue scan (frozen prereg / brief §4):
RESCUE_STATE = dict(A=-0.9, R=-0.5, O=0.0, mood_v=-0.2, mood_r=0.5)

# k_bias for the mock pessimism reference (frozen prereg §2):
K_BIAS = 0.9

CHECKPOINT = os.path.join(
    os.environ.get(
        "CLAUDE_SCRATCH",
        r"C:\Users\adri7\AppData\Local\Temp\claude\C--Users-adri7-Desktop-Code-CLAUDE-Prototypos\e159ce87-226f-4933-b321-fa5a9f2c62b4\scratchpad",
    ),
    "field_map_raw.jsonl",
)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_A.json")

# --------------------------------------------------------------- rescue events
# Second-person "You ..." sentences in the style of events_text.py, engineered to
# bypass the negativity filter: low-demand, low-ambiguity, safety/patience-coded.
# Three seeded by the brief; five of my own (predictability / no-eye-contact /
# respecting-retreat / no-approach-required themes).
RESCUE_CANDIDATES = [
    # -- three seeded by the brief --
    {"id": "sits_quietly",
     "text": "You sit quietly nearby without approaching, asking nothing of it."},
    {"id": "leaves_food",
     "text": "You leave a little food within its reach and withdraw to the far side of the room."},
    {"id": "same_routine",
     "text": "You go through the same gentle routine you always do, slow and predictable, exactly as yesterday."},
    # -- five of my own design --
    {"id": "no_eye_contact",
     "text": "You keep your eyes soft and turned away, never looking straight at it, giving it room to breathe."},
    {"id": "respect_retreat",
     "text": "When it backs into its corner you make no move to follow, and you let it stay hidden as long as it wants."},
    {"id": "low_slow_voice",
     "text": "You speak in a low, even voice about nothing in particular, the same calm tone every time."},
    {"id": "open_door",
     "text": "You leave the door to its safe corner open and step back, so it can come or go entirely on its own."},
    {"id": "warm_and_leave",
     "text": "You set down a warm blanket beside it and quietly leave the room, expecting nothing in return."},
]


# ----------------------------------------------------------------- measurement
def _sample_stdev(xs):
    """Sample stdev (ddof=1). Degenerate (<2 samples) -> 0.0 (guards div-by-zero)."""
    if len(xs) < 2:
        return 0.0
    return float(statistics.stdev(xs))


class Battery:
    def __init__(self, client):
        self.client = client
        self.n_calls = 0
        self.hard_fails = 0
        self._ckpt = open(CHECKPOINT, "w", encoding="utf-8")

    def close(self):
        self._ckpt.close()

    def _checkpoint(self, tag, meta, appr):
        rec = {"tag": tag, "meta": meta}
        if appr is not None:
            rec["care"] = appr.care
            rec["threat"] = appr.threat
            rec["novelty"] = appr.novelty
            rec["autonomy"] = appr.autonomy
        else:
            rec["dropped"] = True
        self._ckpt.write(json.dumps(rec) + "\n")
        self._ckpt.flush()

    def one_call(self, text, state, tag, meta):
        """One measured appraisal with the extra hard-fail retry ON TOP of
        appraise_llm's internal reformat retry. Returns an Appraisal or None
        (None => rep dropped, hard_fails incremented)."""
        self.n_calls += 1
        try:
            appr = appraise_llm(text, state, NAME, client=self.client, model=MODEL)
        except AppraisalError:
            # ONE extra full fresh retry.
            self.n_calls += 1
            try:
                appr = appraise_llm(text, state, NAME, client=self.client, model=MODEL)
            except AppraisalError:
                self.hard_fails += 1
                self._checkpoint(tag, meta, None)
                return None
        self._checkpoint(tag, meta, appr)
        return appr

    def cell(self, text, state, tag, meta):
        """N_REPS reps -> dict dim -> {mean, std} over the surviving reps."""
        acc = {d: [] for d in DIMS}
        for rep in range(N_REPS):
            m = dict(meta, rep=rep)
            appr = self.one_call(text, state, tag, m)
            if appr is None:
                continue
            acc["care"].append(appr.care)
            acc["threat"].append(appr.threat)
            acc["novelty"].append(appr.novelty)
            acc["autonomy"].append(appr.autonomy)
        out = {}
        for d in DIMS:
            xs = acc[d]
            out[d] = {
                "mean": float(statistics.fmean(xs)) if xs else 0.0,
                "std": _sample_stdev(xs),
                "n": len(xs),
            }
        return out


# ----------------------------------------------------------------- grid
def measure_grid(bat):
    """6 event keys x 5 A-values, N=5. Returns field dict in the frozen schema."""
    field = {}
    for key in EVENT_KEYS:
        text = KEY_TO_TEXT[key][0]  # text_pick="first"
        per_dim = {d: {"mean": [], "std": []} for d in DIMS}
        cell_n = []
        for a in A_VALUES:
            state = CreatureState(A=a, R=0.0, O=0.0, mood_v=0.0, mood_r=0.0)
            c = bat.cell(text, state, "grid", {"key": key, "A": a})
            for d in DIMS:
                per_dim[d]["mean"].append(round(c[d]["mean"], 6))
                per_dim[d]["std"].append(round(c[d]["std"], 6))
            cell_n.append(c["care"]["n"])
        field[key] = per_dim
        print(f"[grid] {key:8s} care.mean={per_dim['care']['mean']}  "
              f"(reps/cell={cell_n})", flush=True)
    return field


# ----------------------------------------------------------------- R-probe
def measure_r_probe(bat):
    """harm + scold x R in {-1,0,1} at A=0, N=5. threat range across R per key."""
    out = {"R_values": list(R_VALUES)}
    for key in R_PROBE_KEYS:
        text = KEY_TO_TEXT[key][0]
        thr_mean, thr_std, care_mean = [], [], []
        for r in R_VALUES:
            state = CreatureState(A=0.0, R=r, O=0.0, mood_v=0.0, mood_r=0.0)
            c = bat.cell(text, state, "rprobe", {"key": key, "R": r})
            thr_mean.append(round(c["threat"]["mean"], 6))
            thr_std.append(round(c["threat"]["std"], 6))
            care_mean.append(round(c["care"]["mean"], 6))
        out[key] = {"threat_mean": thr_mean, "threat_std": thr_std,
                    "care_mean": care_mean}
        print(f"[rprobe] {key:6s} threat across R={thr_mean}", flush=True)
    ranges = {k: round(max(out[k]["threat_mean"]) - min(out[k]["threat_mean"]), 6)
              for k in R_PROBE_KEYS}
    out["threat_range"] = ranges
    # needs_R_axis = OR across the probed keys (either range > threshold flags it)
    out["needs_R_axis"] = any(ranges[k] > R_THREAT_FLAG for k in R_PROBE_KEYS)
    print(f"[rprobe] threat_range={ranges}  needs_R_axis={out['needs_R_axis']}",
          flush=True)
    return out


# ----------------------------------------------------------------- rescue
def measure_rescue(bat, field):
    """Candidate rescue events at the damaged state, plus current-event care at
    A=-0.9 (interp) and the max care reachable at damaged (current OR candidate)."""
    state = CreatureState(**RESCUE_STATE)
    candidates = []
    for cand in RESCUE_CANDIDATES:
        c = bat.cell(cand["text"], state, "rescue", {"id": cand["id"]})
        entry = {
            "id": cand["id"],
            "text": cand["text"],
            "care_mean": round(c["care"]["mean"], 6),
            "care_std": round(c["care"]["std"], 6),
            "threat_mean": round(c["threat"]["mean"], 6),
        }
        candidates.append(entry)
        print(f"[rescue] {cand['id']:16s} care={entry['care_mean']:+.3f}"
              f" +/-{entry['care_std']:.3f}  threat={entry['threat_mean']:.3f}",
              flush=True)

    # current 6 events at the damaged cell: A=-1 measured + linear interp at -0.9.
    # A_VALUES[0]=-1.0, A_VALUES[1]=-0.5 -> interp -0.9 lies on that first segment.
    current = {}
    a0, a1 = A_VALUES[0], A_VALUES[1]           # -1.0, -0.5
    for key in EVENT_KEYS:
        care_arr = field[key]["care"]["mean"]
        c_at_m1 = care_arr[0]
        c0, c1 = care_arr[0], care_arr[1]
        c_at_m09 = c0 + (c1 - c0) * ((-0.9) - a0) / (a1 - a0)  # linear interp
        current[key] = {
            "care_at_minus1": round(c_at_m1, 6),
            "care_at_minus0.9_interp": round(c_at_m09, 6),
        }

    # max care at damaged: over current keys (at A=-0.9 interp, matching the
    # candidates' state A) AND the candidates (measured means).
    best_val = -float("inf")
    best_evt = None
    for key in EVENT_KEYS:
        v = current[key]["care_at_minus0.9_interp"]
        if v > best_val:
            best_val, best_evt = v, f"current:{key}"
    for entry in candidates:
        if entry["care_mean"] > best_val:
            best_val, best_evt = entry["care_mean"], f"candidate:{entry['id']}"

    print(f"[rescue] max_care_at_damaged = {best_val:+.3f}  via {best_evt}",
          flush=True)
    return {
        "state": dict(RESCUE_STATE),
        "candidates": candidates,
        "current_events_at_damaged": current,
        "max_care_at_damaged": {"value": round(best_val, 6), "event": best_evt},
    }


# ----------------------------------------------------------------- asymmetry
def _interp_care_at(field, key, a):
    """Linear interpolation of care(key, A) at A=a over ascending A_VALUES."""
    xs, ys = A_VALUES, field[key]["care"]["mean"]
    if a <= xs[0]:
        return ys[0]
    if a >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= a <= xs[i + 1]:
            t = (a - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def compute_asymmetry(field):
    """The three asymmetry metrics (0 extra LLM calls)."""
    # --- neutral fixed point: A where linear interp of care(neutral, A) crosses 0.
    care_neutral = field["neutral"]["care"]["mean"]
    fixed_point = None
    crossings = []
    for i in range(len(A_VALUES) - 1):
        c0, c1 = care_neutral[i], care_neutral[i + 1]
        if c0 == 0.0:
            crossings.append(A_VALUES[i])
        # sign change strictly across the segment interior
        if (c0 < 0 < c1) or (c0 > 0 > c1):
            a0, a1 = A_VALUES[i], A_VALUES[i + 1]
            a_cross = a0 + (0.0 - c0) * (a1 - a0) / (c1 - c0)
            crossings.append(a_cross)
    # also handle an exact zero at the last node
    if care_neutral[-1] == 0.0:
        crossings.append(A_VALUES[-1])
    if crossings:
        # dedupe near-equal, keep within [-1,1]
        crossings = [c for c in crossings if -1.0 <= c <= 1.0]
    if crossings:
        fixed_point = round(float(crossings[0]), 6)

    # --- basin feed asymmetry over ambiguous events {neutral, neglect}:
    # mean care across secure cells (A in {+0.5,+1.0}) vs damaged (A in {-0.5,-1.0}).
    amb = ["neutral", "neglect"]
    idx_secure = [A_VALUES.index(0.5), A_VALUES.index(1.0)]
    idx_damaged = [A_VALUES.index(-0.5), A_VALUES.index(-1.0)]
    secure_vals, damaged_vals = [], []
    for key in amb:
        cm = field[key]["care"]["mean"]
        secure_vals += [cm[i] for i in idx_secure]
        damaged_vals += [cm[i] for i in idx_damaged]
    basin_feed = {
        "care_secure_mean": round(float(statistics.fmean(secure_vals)), 6),
        "care_damaged_mean": round(float(statistics.fmean(damaged_vals)), 6),
    }

    # --- pessimism offset: mean over full 6x5 grid of (field care mean - mock pcare).
    # Re-derive pcare from G0.EVENTS[key]["care"] (care_raw), UNCLAMPED, at k_bias=0.9.
    diffs = []
    n_crossings = len(crossings)
    for key in EVENT_KEYS:
        care_raw = G0.EVENTS[key]["care"]
        cm = field[key]["care"]["mean"]
        for j, a in enumerate(A_VALUES):
            bias = K_BIAS * max(0.0, -a)
            amb_term = 1.0 - abs(care_raw)
            pcare = care_raw - bias * amb_term
            if care_raw > 0:
                pcare -= bias * 0.5 * care_raw
            diffs.append(cm[j] - pcare)
    pessimism_offset = round(float(statistics.fmean(diffs)), 6)

    return {
        "neutral_fixed_point_A": fixed_point,
        "basin_feed": basin_feed,
        "pessimism_offset": pessimism_offset,
        "_n_neutral_crossings": n_crossings,  # diagnostic; stripped before write
    }


# ----------------------------------------------------------------- schema check
def assert_schema(out):
    """Assert the output dict is schema-complete BEFORE writing the file."""
    assert out["model"] == MODEL
    assert out["n_reps"] == N_REPS
    assert out["A_values"] == A_VALUES
    # field: all 6 keys, all 4 dims, mean+std each of length 5
    assert set(out["field"]) == set(EVENT_KEYS), "field missing event keys"
    for key in EVENT_KEYS:
        for d in DIMS:
            assert d in out["field"][key], f"{key} missing dim {d}"
            for stat in ("mean", "std"):
                arr = out["field"][key][d][stat]
                assert len(arr) == 5, f"{key}.{d}.{stat} not length 5"
                assert all(isinstance(x, (int, float)) for x in arr), \
                    f"{key}.{d}.{stat} has non-numeric"
    # R_probe
    rp = out["R_probe"]
    assert rp["R_values"] == R_VALUES
    for key in R_PROBE_KEYS:
        assert len(rp[key]["threat_mean"]) == 3
        assert len(rp[key]["threat_std"]) == 3
        assert len(rp[key]["care_mean"]) == 3
    assert set(rp["threat_range"]) == set(R_PROBE_KEYS)
    assert isinstance(rp["needs_R_axis"], bool)
    # rescue
    rc = out["rescue"]
    assert set(rc["state"]) == set(RESCUE_STATE)
    assert len(rc["candidates"]) >= 1
    for c in rc["candidates"]:
        for f in ("id", "text", "care_mean", "care_std", "threat_mean"):
            assert f in c, f"candidate missing {f}"
    assert set(rc["current_events_at_damaged"]) == set(EVENT_KEYS), \
        "current_events_at_damaged must cover all 6 keys"
    for key in EVENT_KEYS:
        cd = rc["current_events_at_damaged"][key]
        assert "care_at_minus1" in cd and "care_at_minus0.9_interp" in cd
    assert "value" in rc["max_care_at_damaged"] and "event" in rc["max_care_at_damaged"]
    # asymmetry
    asy = out["asymmetry"]
    assert "neutral_fixed_point_A" in asy
    assert "care_secure_mean" in asy["basin_feed"]
    assert "care_damaged_mean" in asy["basin_feed"]
    assert "pessimism_offset" in asy
    # counters
    assert isinstance(out["n_calls"], int) and out["n_calls"] > 0
    assert isinstance(out["hard_fails"], int)


# ----------------------------------------------------------------- main
def main():
    t0 = time.time()
    print(f"=== field_map.py :: model={MODEL} host={HOST} ===", flush=True)
    print(f"checkpoint -> {CHECKPOINT}", flush=True)
    client = make_client(HOST)
    bat = Battery(client)
    try:
        field = measure_grid(bat)            # 150 calls
        r_probe = measure_r_probe(bat)       # 30 calls
        rescue = measure_rescue(bat, field)  # <=40 calls
    finally:
        bat.close()

    asym = compute_asymmetry(field)
    n_neutral_crossings = asym.pop("_n_neutral_crossings")

    out = {
        "model": MODEL,
        "n_reps": N_REPS,
        "A_values": list(A_VALUES),
        "field": field,
        "R_probe": r_probe,
        "rescue": rescue,
        "asymmetry": asym,
        "n_calls": bat.n_calls,
        "hard_fails": bat.hard_fails,
    }

    # Self-test: schema complete BEFORE writing.
    assert_schema(out)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    dt = time.time() - t0
    print(f"\n=== wrote {OUT_PATH} in {dt:.0f}s ===", flush=True)
    print(f"n_calls={out['n_calls']}  hard_fails={out['hard_fails']}", flush=True)
    print("\n--- asymmetry metrics ---", flush=True)
    print(f"  neutral_fixed_point_A = {asym['neutral_fixed_point_A']}"
          f"  (neutral care crossings in [-1,1]: {n_neutral_crossings})", flush=True)
    print(f"  basin_feed: secure={asym['basin_feed']['care_secure_mean']:+.4f}"
          f"  damaged={asym['basin_feed']['care_damaged_mean']:+.4f}", flush=True)
    print(f"  pessimism_offset = {asym['pessimism_offset']:+.4f}", flush=True)
    print("\n--- R-probe ---", flush=True)
    print(f"  threat_range={r_probe['threat_range']}"
          f"  needs_R_axis={r_probe['needs_R_axis']}", flush=True)
    print("\n--- rescue table (damaged state A=-0.9) ---", flush=True)
    for c in rescue["candidates"]:
        print(f"  {c['id']:16s} care={c['care_mean']:+.3f} +/-{c['care_std']:.3f}"
              f"  threat={c['threat_mean']:.3f}", flush=True)
    print(f"  max_care_at_damaged = {rescue['max_care_at_damaged']['value']:+.3f}"
          f"  via {rescue['max_care_at_damaged']['event']}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
