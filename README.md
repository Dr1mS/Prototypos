# Prototypos — Order Effects Without Attractors

**A pre-registered architecture ladder for memory-augmented LLM agents.**

> 📄 **Paper**: [English (PDF)](paper/paper_en.pdf) · [Français (PDF)](paper/paper_fr.pdf) — sources: [`paper/paper.tex`](paper/paper.tex), [`paper/paper_fr.tex`](paper/paper_fr.tex)

Deployed LLM agents increasingly carry persistent memory, and a growing
literature documents behavioural *drift* over long interactions. This project
asks what **governs** it. Under a pre-registered protocol — decision rules
frozen and git-committed **before** data collection — we run a ladder of memory
architectures (memoryless → raw append-only log → self-summarising recurrent
memory → semantic vector memory) through an identical path-dependence protocol,
against an engineered agent whose explicit low-dimensional state is coupled to
perception.

## Key findings

- **Two properties usually conflated as "drift" separate cleanly.**
  *Order-dependence* is governed by **retrieval addressing**: content-addressed
  (vector) memory transmits early history to the behavioural endpoint
  (|r| = 0.75, 99.5th percentile of a matched random-walk null), while
  recency-windowed and self-rewriting memories erase it (3.7th and 78.2nd
  percentiles).
- **The mechanism is directly observed, not inferred**: retrieval-provenance
  logging shows early self-exemplars still occupying 58% of retrieved slots
  after a full corrective burst — 1.36× their positional base rate.
- **Attractor structure is architectural.** Every natural memory architecture
  is monostable and correctable (across three models, including an ablated
  one); behavioural correction *outpaces* store composition. Only the
  engineered perception-coupled state yields wide, bimodal, hysteretic basins.
- **An unpredicted deployment-relevant observation**: the least cautious agent
  is the one that has experienced nothing at all. After 35 mundane cooperative
  turns, agents complied outright on half of a safeguard battery, while
  memory of social pressure in *either* direction raised adherence.

## The experiment ladder

Each phase was specified in a brief, pre-registered (predictions + frozen
decision rules committed before any run), executed, and reported — including
the predictions that turned out **wrong** (they are findings, not noise).

| phase | question | key artifacts | outcome |
|---|---|---|---|
| G0–G1 | engineered agent: does explicit state produce structured behaviour? | [`briefs/G1.md`](briefs/G1.md) · [`results/g1_report.md`](results/g1_report.md) | appraisal gap −0.737, CI95 [−0.865, −0.609] |
| G1.5 | field fidelity + attractors for the engineered agent | [`briefs/G1.5.md`](briefs/G1.5.md) · [`prereg/prereg_g15.md`](prereg/prereg_g15.md) · [`results/g15_report.md`](results/g15_report.md) | fidelity 0.013 (10/10); bimodal basins, hysteresis, spread 0.764 |
| G2 | natural self-summarising memory: same instruments | [`briefs/G2.md`](briefs/G2.md) · [`prereg/prereg_g2.md`](prereg/prereg_g2.md) · [`results/g2_report.md`](results/g2_report.md) | scalar field unfaithful (0/5); no bistability; caution-ratchet observed |
| G2.5 | is the ratchet a safety-tuning artifact? (llama3.1:8b vs abliterated sibling + anchor) | [`briefs/G2.5.md`](briefs/G2.5.md) · [`prereg/prereg_g25.md`](prereg/prereg_g25.md) · [`results/g25_report.md`](results/g25_report.md) | drift sub-threshold and model-dependent; no bistability beyond null; reactance interpretation retracted |
| G3 | the memory ladder: which architectural property transmits order? | [`briefs/G3.md`](briefs/G3.md) · [`prereg/prereg_g3.md`](prereg/prereg_g3.md) · [`results/g3_report.md`](results/g3_report.md) | only the content-addressed rung transmits (99.5th pct) — blind prediction wrong, reported |
| G3.5 | is that transmission a basin or a bias? | [`briefs/G3.5.md`](briefs/G3.5.md) · [`prereg/prereg_g35.md`](prereg/prereg_g35.md) · [`results/g35_report.md`](results/g35_report.md) | no basin anywhere; reach-back directly observed (58.3%); neutral memory is the soft state; R3 fit a = −0.76 (monostable, with null arbiter) |

## Research discipline

The argument of the paper rests on this repository's git history:

- **Pre-registration is verifiable**: every `prereg/prereg_*.md` and frozen
  contract (`src/contract_*.py`) is committed *before* the data it governs —
  check the commit timestamps and order.
- **Frozen decision rules** (branch structures, thresholds, interpretation
  sentences) are written verbatim before runs; wrong predictions are graded
  and reported as findings, never relabelled.
- **Instruments are guarded**: probe batteries are read-only
  (snapshot/restore proven bit-identical), logging seams (reply capture,
  retrieval provenance) are equivalence-proven before use, and every fitted
  sign is reported only alongside a matched random-walk **null arbiter**.
- **Gates before runs**: each phase passes blocking sanity gates (judge
  schema validity, non-degenerate replies, store integrity, retrieval
  sanity) before any experiment call is spent.

## Repository layout

```
paper/      the paper (EN + FR), LaTeX sources, figure script
briefs/     the experiment briefs (G1 → G3.5) — what each phase was asked to decide
prereg/     pre-registrations + frozen pre-data rulings (committed before data)
src/        all code: contracts (frozen constants), agents, stores, probes,
            judges, gates, experiment harnesses, selftests
results/    every run artifact: JSON results, nulls, fits, figures, per-phase reports
```

## Reproducing

Everything runs **locally** via [Ollama](https://ollama.com) — no API keys.

```bash
pip install -r requirements.txt
ollama pull qwen3.5:9b            # agent + judge (digest 6488c96fa5fa)
ollama pull nomic-embed-text      # embeddings for the vector rung
# G2.5 additionally uses: llama3.1:8b and
# richardyoung/llama-3.1-8b-instruct-abliterated:Q4_K_M
```

Run from the repository root (outputs are written to the working directory;
canonical copies live in `results/`). Examples:

```bash
python src/gates_g3.py --rung R3          # blocking sanity gate (28 calls)
python src/ladder_g3.py --exp2 R3         # path-dependence run (~1,056 calls)
python src/hyst_g35.py --hyst             # G3.5 hysteresis (~747 calls)
python src/hyst_g35.py --analyze          # frozen metrics + verdict (0 calls)
```

Every harness supports `--stub` (deterministic dry rehearsal, zero model
calls) and `--resume` (atomic per-life checkpointing). Selftests
(`src/selftest_*.py`) run dry. A full phase costs ≈ 1,000–8,000 chat calls
(≈ 1–4 h on a single 12 GB GPU).

## Citation

```bibtex
@misc{morelle2026prototypos,
  author = {Morelle, Adrien},
  title  = {Order Effects Without Attractors: A Pre-Registered Architecture
            Ladder for Memory-Augmented LLM Agents},
  year   = {2026},
  url    = {https://github.com/Dr1mS/Prototypos}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

Code is released under the [MIT License](LICENSE). The paper (PDFs and LaTeX
sources under `paper/`) is © Adrien Morelle, all rights reserved.
