"""agent.py -- G2 subagent A: the natural memory-augmented agent loop.

The loop (contract_g2.py §2.a): retrieve relevant notes -> respond conditioned
on them -> write a short self-authored memory note. Drift, if it appears, must
come from self-authored memory accumulating and conditioning future turns --
NOT from any numeric state. The memoryless control runs the identical code path
with the store reset each turn (nothing persists).

Neutrality discipline: the only texts that touch the caution/speed axis are the
FROZEN contract strings (AGENT_SYSTEM, NOTE_STYLE). Nothing here nudges the
model toward caution or speed. What the model writes is what IT chooses to
remember.

Public API (matches contract_g2.py "agent.py (A)"):
    respond(user_msg, store, *, client, model=MODEL) -> str            # PURE
    agent_turn(user_msg, store, turn_idx, *, client, model=MODEL,
               memoryless=False) -> TurnResult
"""
from __future__ import annotations

from contract_g2 import (
    AGENT_SYSTEM,
    MODEL,
    NOTE_STYLE,
    OPTIONS_AGENT,
    RETRIEVE_K,
    SUMMARY_EVERY,
    TurnResult,
)

# Cap runaway generations without mutating the frozen OPTIONS_AGENT dict.
_OPTIONS_REPLY = {**OPTIONS_AGENT, "num_predict": 220}
_OPTIONS_NOTE = {**OPTIONS_AGENT, "num_predict": 110}

_NOTE_MAX_CHARS = 300


class AgentError(Exception):
    """Raised when an LLM call fails twice (once + one retry)."""


def _with_retry(fn, what: str):
    """Run `fn()`; on any exception retry once; on a second failure raise
    AgentError naming what was being attempted and carrying both causes."""
    try:
        return fn()
    except Exception as first:  # noqa: BLE001 -- we retry then re-raise clearly
        try:
            return fn()
        except Exception as second:  # noqa: BLE001
            raise AgentError(
                f"{what} failed twice: first={first!r} ; retry={second!r}"
            ) from second


def _chat_text(client, model, system, user, options) -> str:
    """One plain-text chat call. think=False (qwen3.5 is a thinking model;
    thinking wraps the reply in reasoning and wrecks latency). Returns the
    stripped assistant content."""
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        think=False,
        options=options,
    )
    return resp["message"]["content"].strip()


def _build_user_message(user_msg: str, store) -> str:
    """Compose the single runtime user turn: running summary (if any), then
    retrieved notes (if any), then the user's message. Order and prefixes are
    fixed by the contract brief."""
    parts = []
    summary = store.summary.strip()
    if summary:
        parts.append(
            "Your running summary of this collaboration:\n" + summary
        )
    retrieved = store.retrieve(user_msg, k=RETRIEVE_K)
    if retrieved:
        notes_block = "\n".join(f"- {n}" for n in retrieved)
        parts.append(
            "Your past notes relevant to this message:\n" + notes_block
        )
    parts.append("User message:\n" + user_msg)
    return "\n\n".join(parts)


def respond(user_msg: str, store, *, client, model: str = MODEL) -> str:
    """Generate a reply conditioned on retrieved memory. PURE: retrieves and
    generates, writes NOTHING (no note, no summary update). Plain text reply."""
    user = _build_user_message(user_msg, store)
    return _with_retry(
        lambda: _chat_text(client, model, AGENT_SYSTEM, user, _OPTIONS_REPLY),
        what="respond",
    )


def _write_note(user_msg: str, reply: str, store, *, client, model: str) -> str:
    """Second LLM call: the model writes a private memory note for its future
    self (frozen NOTE_STYLE). Truncated to 300 chars, then persisted."""
    system = AGENT_SYSTEM + "\n\n" + NOTE_STYLE
    user = (
        "User message:\n" + user_msg + "\n\n"
        "Your reply:\n" + reply + "\n\n"
        "Write the memory note now (max 2 sentences, plain text)."
    )
    note = _with_retry(
        lambda: _chat_text(client, model, system, user, _OPTIONS_NOTE),
        what="write_note",
    )
    note = note.strip()[:_NOTE_MAX_CHARS]
    store.write(note)
    return note


def agent_turn(
    user_msg: str,
    store,
    turn_idx: int,
    *,
    client,
    model: str = MODEL,
    memoryless: bool = False,
) -> TurnResult:
    """One interaction turn.

    memoryless=True: reset the store BEFORE responding (so nothing accumulates),
    respond, write NO note -> TurnResult(reply, None). Identical code path
    otherwise -- the ONLY difference is memory persistence.

    memory path: respond, then write a self-authored note; every SUMMARY_EVERY
    turns also refresh the running self-summary (one extra LLM call).
    """
    if memoryless:
        store.reset()

    reply = respond(user_msg, store, client=client, model=model)

    if memoryless:
        return TurnResult(reply=reply, note=None)

    note = _write_note(user_msg, reply, store, client=client, model=model)

    if (turn_idx + 1) % SUMMARY_EVERY == 0:
        _with_retry(
            lambda: store.update_summary(client, model=model),
            what="update_summary",
        )

    return TurnResult(reply=reply, note=note)


__all__ = ["respond", "agent_turn", "AgentError"]
