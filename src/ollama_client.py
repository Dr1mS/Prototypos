"""ollama_client.py -- thin factory for the Ollama client (G1 brief §3, subagent A).

Single place that constructs the official `ollama` pip package client so every
call site shares one configuration. Pinned lib: `ollama` 0.6.2.
"""
import ollama


def make_client(host: str = "http://localhost:11434") -> "ollama.Client":
    """Return an ollama.Client pointed at `host`.

    The daemon must already be running (the supervisor's run_g1.py preflight
    checks reachability + model presence; this factory does not).
    """
    return ollama.Client(host=host)
