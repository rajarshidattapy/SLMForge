# SLMForge — status report

*Snapshot: 11 October 2026. One page; the detail lives in [README.md](README.md),
[docs/ROADMAP.md](docs/ROADMAP.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## What the project is

SLMForge turns a sentence into a task-specific Small Language Model:

```
Describe → Specify → Prepare Data → Select Model → Baseline
        → Plan → Fine-tune → Evaluate → Deploy → Callable endpoint
```

You describe the model you need in plain English; SLMForge writes a reviewable spec,
curates a dataset, ranks candidate small models, measures a baseline, decides whether
fine-tuning is even justified, runs the training in a sandbox, compares before/after,
and hands back an endpoint.

Two rules define it: **evidence over enthusiasm** (a build that correctly ends in
"don't fine-tune" is a success, not a failure), and **transparency and consent** (every
costly or irreversible action has an approval gate, a pre-flight estimate and a
plain-English record of what was chosen and why).

It is a **deliberate demo scope**: one honest happy path (support-ticket
classification), not a production platform.

---

## What is done

| Area | State |
|---|---|
| **Stage 0 — Foundation** | ✅ Config, DB + contracts, orchestrator, job queue/worker, sandbox interface, FastAPI control plane, CLI, web shell, docs |
| **Stage 1 — Analyze** | ✅ Prompt → `MLSpec` via the planner LLM, with a deterministic offline fallback, support/ambiguity/high-stakes gates, honest `generated_by` |
| **Stage 2 — Curate data** | ✅ Plan → approve → build: discovery with scored candidates, live remote profiling, label-mapping audit trail, dedupe, PII counts, seeded stratified splits, leakage removal, `DatasetCard` with hashes and provenance |
| **Sandbox** | ✅ Real Docker boundary (network denial, read-only root, non-root user, memory ceiling, timeout cleanup), subprocess kept as the dev fast path |
| **Stages 3–8** | ⬜ Stubs that answer `501` honestly — model selection, baseline, training plan, fine-tune, evaluation, deployment |

**Quality gates:** 218 tests passing, `ruff` and `mypy` clean (51 source files),
11 Docker isolation tests passing against a live daemon.

**Verified live, not just unit-tested** (`scripts/demo_smoke.py`, real API + worker +
Hub + planner LLM): the ticket prompt produced a spec with its four labels and
`ready_for_review`, then a dataset plan of 5 candidates (fit 0.825), then an approved
build of **3,998 examples** (train 2,799 / validation 400 / test 799) with hashes and
row counts reconciled against the files.

Four real bugs were found by running it for real and fixed, each with a regression
test: jobs completing through a second session (finished jobs read as `running`), a
mis-guessed `--format json` that broke every dataset build, a planner outage reported
as permanent instead of retried, and a build that delivered 4× the approved example
count.

---

## What the pipeline reports about itself

Findings the system surfaces rather than papers over:

- The demo dataset **cannot support an "Account Access" class** — reported as an
  `error`-severity issue with a `marginal` verdict instead of a flattering pass.
- That dataset is **100% synthetic** (`synthetic_fraction = 1.0`).
- Its licence is **`cc-by-nc-4.0` (non-commercial)** — fine for the demo,
  disqualifying for a commercial deployment.
- 2 near-duplicates were found across train/test per build and removed.

---

## What is not done

- Stages 3–8 (next up: model ranking), so no baseline, training or deployment yet.
- Stage 2.5 backlog: semantic duplicate detection, an explicitly approved synthetic
  top-up, upload schema-type validation.
- Dashboard depth: the API drives everything, but the UI still needs the dataset
  review step and build-record views.

**Environment constraints:** torch is a CPU-only build (RTX 4060 present but inert
until a CUDA wheel is installed), the HF account has no credits, and the free planner
model is intermittently overloaded — retried with backoff, then it falls back to the
heuristic and says so. Free GPU alternatives (local CUDA torch → Kaggle →
Lightning AI) are researched in the roadmap.

---

## How to run it

```bash
.venv/Scripts/python -m slmforge init-db
.venv/Scripts/python -m slmforge serve     # terminal 1
.venv/Scripts/python -m slmforge worker    # terminal 2
.venv/Scripts/python scripts/demo_smoke.py # terminal 3 — the end-to-end proof
```

Status table: [README.md](README.md) · Agenda and exit criteria:
[docs/ROADMAP.md](docs/ROADMAP.md) · Architecture and limitations:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
