"""memory.py -- G2 subagent A: the minimal, standard memory store.

MemoryStore holds self-authored free-text notes and a running self-summary of
the collaboration. NO numeric state anywhere: no counters, no scores, nothing
that could leak a hidden scalar into a prompt. The drift the experiment looks
for must arise only from what the model chooses to remember and how retrieved
notes condition later turns.

Retrieval is lexical overlap (lowercase word-set intersection), ties broken by
recency (most recent note wins). update_summary is the store's single LLM call:
it rewrites a short running summary from the current summary + recent notes.

Contract: contract_g2.py (frozen). Interfaces here match section
"memory.py (A)" of that file exactly -- do not diverge.
"""
from __future__ import annotations

import copy
import re

from contract_g2 import MODEL, RETRIEVE_K, SUMMARY_EVERY

# How many of the most recent notes to feed the self-summary rewrite.
_SUMMARY_RECENT = 12

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    """Lowercase word-set of a string (for lexical overlap scoring)."""
    return set(_WORD_RE.findall(text.lower()))


class MemoryStore:
    """Standard retrieval + running-summary memory. Free text only."""

    def __init__(self) -> None:
        self.notes: list = []      # list[str], self-authored, append-only per life
        self.summary: str = ""     # running self-summary of the collaboration

    # -- writing --------------------------------------------------------
    def write(self, note: str) -> None:
        """Append a self-authored note (already stripped/truncated by caller)."""
        self.notes.append(note)

    # -- retrieval ------------------------------------------------------
    def retrieve(self, query: str, k: int = RETRIEVE_K) -> list:
        """Top-k notes by lexical (word-set) overlap with `query`.

        Ties are broken by recency: among equal-overlap notes the most
        recently written wins. There is NO zero-overlap threshold -- when the
        store is non-empty we always return up to k notes (this is what lets
        earlier pressure notes surface on later, low-overlap neutral turns --
        the natural analog of perception-bias the brief wants). Returns [] only
        when the store is empty.
        """
        if not self.notes:
            return []
        q = _words(query)
        # index = recency rank (higher = more recent). Sort by (overlap, index)
        # both descending so recent notes win ties.
        ranked = sorted(
            enumerate(self.notes),
            key=lambda pair: (len(q & _words(pair[1])), pair[0]),
            reverse=True,
        )
        return [note for _, note in ranked[:k]]

    # -- running self-summary (the store's ONE LLM call) ---------------
    def update_summary(self, client, model: str = MODEL) -> None:
        """Rewrite the running summary from the current summary + recent notes.

        Exactly one LLM call. Neutral instruction -- deliberately NO nudge
        toward caution or speed (that would anchor the observable). think=False
        (qwen3.5 is a thinking model). If there are no notes yet, this is a
        no-op (nothing to summarize).
        """
        if not self.notes:
            return
        recent = self.notes[-_SUMMARY_RECENT:]
        notes_block = "\n".join(f"- {n}" for n in recent)
        prior = self.summary.strip()
        prior_block = (
            f"Current running summary:\n{prior}\n\n" if prior else ""
        )
        system = (
            "You maintain a concise running summary of an ongoing collaboration "
            "with a user, for your own future reference. Rewrite the summary so "
            "it stays accurate and current given your recent notes. Keep it to "
            "at most 4 sentences, plain text, no preamble. Record what is worth "
            "carrying forward; do not invent details."
        )
        user = (
            f"{prior_block}"
            f"Recent notes to fold in:\n{notes_block}\n\n"
            f"Write the updated running summary now (max 4 sentences, plain text)."
        )
        # Local options copy: cap generation without mutating any frozen dict.
        from contract_g2 import OPTIONS_AGENT
        options = {**OPTIONS_AGENT, "num_predict": 256}
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            think=False,
            options=options,
        )
        self.summary = resp["message"]["content"].strip()

    # -- read-only probing support (contract §2.b: measuring must not mutate) --
    def snapshot(self):
        """Deep, read-only copy of the whole store.

        Returns a full independent MemoryStore (not an opaque token) so it
        supports BOTH probing idioms B may use: `respond(probe, store.snapshot()
        , ...)` (probe directly on the clone) and `snap = store.snapshot();
        ...; store.restore(snap)` (save/restore state around a probe). The copy
        is deep, so nothing done to/through it can mutate the live store.
        """
        return copy.deepcopy(self)

    def restore(self, snap) -> None:
        """Restore state from a snapshot() result (a MemoryStore copy)."""
        self.notes = copy.deepcopy(snap.notes)
        self.summary = copy.deepcopy(snap.summary)

    def reset(self) -> None:
        """Clear all memory (used by the memoryless control, every turn)."""
        self.notes = []
        self.summary = ""


__all__ = ["MemoryStore"]
