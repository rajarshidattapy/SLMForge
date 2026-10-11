# SLMForge

**Turn a natural-language requirement into a task-specific Small Language Model.**

Describe the model you need in plain English. SLMForge interprets the request,
curates a dataset, finds and ranks candidate small models, measures a baseline,
decides *whether* fine-tuning is even justified, runs the training inside a
sandbox, compares the result against the baseline, and hands back a **callable
OpenAI-compatible endpoint**.

```
Describe → Specify → Prepare Data → Select Model → Baseline
        → Plan → Fine-tune → Evaluate → Deploy → Callable endpoint
```

> **Project scope: demo, deliberately.** This repository builds a *demonstrable*
> end-to-end prototype on one happy path (support-ticket classification), not a
> production multi-tenant platform. "Works only up to the demo state" is a design
> decision: every stage must produce verifiable evidence on a real run before the
> next one starts. See [Scope and non-goals](#scope-and-non-goals).

> **Just want the two-minute version?** [REPORT.md](REPORT.md) is a one-page summary
> of what SLMForge is and what has been built so far.

---

## Table of contents

1. [Current state](#current-state) — what actually works today
2. [Quick start](#quick-start) — from clone to running API in five minutes
3. [Running the demo path](#running-the-demo-path)
4. [Configuration](#configuration)
5. [Repository layout](#repository-layout)
6. [Architecture](#architecture)
7. [Testing and quality gates](#testing-and-quality-gates)
8. [Contributing](#contributing)
9. [Troubleshooting](#troubleshooting)
10. [Scope and non-goals](#scope-and-non-goals)
11. [Licence](#licence)

---

## Current state

As of the last commit on `main`:

| Check | Result |
|---|---|
| `pytest` | **218 passing**, 0 failing |
| `ruff check src tests` | clean |
| `ruff format` | clean |
| `mypy src` | clean (51 source files) |
| Docker sandbox isolation tests | 11 passing against a live daemon |
| End-to-end demo (prompt → dataset) | verified live against the real Hub **and** the real planner LLM |

### Stage-by-stage status

| # | Stage | Status | Notes |
|---|---|---|---|
| 0 | Foundation | ✅ Complete | Config, DB, contracts, orchestrator, queue/worker, sandbox interface, API, CLI, web shell, docs |
| 1 | Analyze → `MLSpec` | ✅ Complete | Planner-LLM path **and** deterministic offline fallback; high-stakes + support gates; clarifying questions |
| 2 | Curate data | ✅ Complete | Plan → approve → build: label mapping, dedupe, seeded splits, leakage check, `DatasetCard`, uploaded files |
| 3 | Select model | ⬜ **Next up** | |
| 4 | Sandbox + baseline | 🟡 **Docker sandbox done** | Image built, isolation proven; baseline harness pending |
| 5 | Training planner | ⬜ Not started | |
| 6 | Fine-tune | ⬜ Not started | |
| 7 | Evaluate | ⬜ Not started | |
| 8 | Deploy | ⬜ Not started | |
| 9 | Dashboard & hardening | ⬜ Not started | |

Legend: ✅ done and verified · 🟡 partially done · ⬜ not started.

### What you can do *today*

- Create a project from a sentence and get a validated, editable `MLSpec`.
- Watch the analyzer choose its path (planner LLM or offline heuristic) and label
  it honestly in `generated_by`; a planner outage is retried and then reported, never
  hidden.
- See high-stakes and out-of-scope requests refused with a reason rather than
  silently accepted.
- Get a reviewable `DatasetPlan` with scored candidates, a live remote profile, a
  licence verdict and a sufficiency call — without downloading anything.
- Approve a build and get real `train/validation/test.jsonl` files plus a
  `DatasetCard`: label mapping audit trail, duplicate and PII counts, leakage removed,
  file hashes, per-source provenance and a synthetic fraction.
- Build a dataset from **your own JSONL/CSV** with a column mapper (`upload:<path>`).
- Queue long work (`POST /api/projects/{id}/jobs`) and let a worker run it.
- Inspect `/api/capabilities` to see exactly what the machine can and cannot do.
- Run untrusted code in a **real Docker sandbox** with network denial, a read-only
  root, a non-root user and CPU/memory/time limits.

### What does *not* work yet

Stages 3–8 (model selection, baseline, training plan, fine-tune, evaluation,
deployment) return an honest `501 Not Implemented` with the milestone that will build
them — by design, so nothing looks finished that isn't. `docs/ROADMAP.md` is the
source of truth for what is next. Stage 2 also carries a short list of deferred work
(semantic duplicate detection, synthetic top-up) under "Stage 2.5".

---

## Quick start

**Prerequisites**

| Requirement | Needed for | Install |
|---|---|---|
| Python ≥ 3.11 | everything | <https://python.org> |
| [`uv`](https://docs.astral.sh/uv/) | environment management | `pip install uv` |
| Node ≥ 20 | the web dashboard | <https://nodejs.org> |
| `hf` CLI, authenticated | datasets, models, Jobs, endpoints | `pip install -U "huggingface_hub[cli]"` then `hf auth login` |
| `duckdb` | remote dataset profiling | `pip install duckdb` (see the note below) |
| Docker Desktop, **running** | the default `docker` sandbox backend | <https://docker.com> |

> **`hf` and `duckdb` install on the *global* interpreter, not into `.venv`.**
> The `hf` CLI shells out to DuckDB and binds to the Python it was installed
> under. Installing `duckdb` into `.venv` will not make it visible to `hf`.

**1. Clone and create the environment**

```bash
git clone <repo-url> SLMForge && cd SLMForge

# Windows (Git Bash)
uv venv --system-site-packages .venv
uv pip install -e '.[dev]' -p .venv/Scripts/python.exe

# macOS / Linux
uv venv --system-site-packages .venv
uv pip install -e '.[dev]' -p .venv/bin/python
```

`--system-site-packages` is intentional on the reference machine: the host already
has torch, transformers, peft, trl and friends, and installing them again would
pull several gigabytes. On a clean machine with room to spare, a plain
`uv venv` plus `uv pip install -e '.[dev,ml]'` also works.

**2. Configure**

```bash
cp .env.example .env      # then edit as needed
```

Nothing in `.env` is strictly required for the analyzer: with no API key the
heuristic path runs and says so. See [Configuration](#configuration).

**3. Check what this machine can do**

```bash
.venv/Scripts/python -m slmforge doctor        # POSIX: .venv/bin/python
```

This reports torch (and whether it has CUDA), the GPU, Ollama, the planner LLM,
the HF CLI, the sandbox backend, and free disk. Fix what it complains about
*before* running a stage — the same information is on `/api/capabilities`.

**4. Build the sandbox image**

```bash
docker build -f docker/sandbox.Dockerfile -t slmforge/sandbox:latest .
```

Skipping this is fine for developing the API, but any dataset or training job
will fail. Not running Docker at all? Set `SLMFORGE_SANDBOX_BACKEND=subprocess` —
and read the caveat: that is process isolation, **not** a security boundary.

**5. Create the database and start the API**

```bash
.venv/Scripts/python -m slmforge init-db
.venv/Scripts/python -m slmforge serve --reload
```

- API docs: <http://127.0.0.1:8000/docs>
- Capabilities: <http://127.0.0.1:8000/api/capabilities>

**6. Start a worker** (second terminal — required for any long-running job)

```bash
.venv/Scripts/python -m slmforge worker
```

**7. Optional: the web dashboard**

```bash
cd web && npm install && npm run dev     # http://localhost:5173
```

> **Use `localhost`, not `127.0.0.1`, for Vite.** On some machines Vite 8 binds
> IPv6 only, so `http://127.0.0.1:5173` is refused while `localhost:5173` works.
> To avoid a second server entirely, run `npm run build` and let FastAPI serve the
> built SPA at <http://127.0.0.1:8000>.

---

## Running the demo path

The supported happy path is **support-ticket classification** into
Billing / Technical / Account Access / General, returning JSON.

```bash
API=http://127.0.0.1:8000

# 1. Create a project from one sentence
PROJECT=$(curl -s -X POST $API/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Build a small model that classifies customer support tickets into Billing, Technical, Account Access or General and returns only a valid JSON object."}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

# 2. Analyze it (or GET /api/projects/$PROJECT/next to see what is due)
curl -s -X POST $API/api/projects/$PROJECT/stages/analyze/run \
  -H 'Content-Type: application/json' -d '{}' | python -m json.tool

# 3. Plan the dataset — profiles candidates remotely, downloads nothing.
#    Expect `needs_approval: true`; the project deliberately does NOT advance.
curl -s -X POST $API/api/projects/$PROJECT/stages/curate_data/run \
  -H 'Content-Type: application/json' -d '{}' | python -m json.tool

# 4. Approve it: a build downloads, so it goes to the worker (a worker must run)
curl -s -X POST $API/api/projects/$PROJECT/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"dataset_build","params":{"build":true,"selected":["hf:Tobi-Bueck/customer-support-tickets"],"target_examples":4000}}'

# 5. Watch it, then read what was built
curl -s $API/api/projects/$PROJECT/jobs     | python -m json.tool
curl -s $API/api/projects/$PROJECT          | python -m json.tool   # dataset_plan.card
curl -s $API/api/projects/$PROJECT/artifacts | python -m json.tool

# Inspect the pipeline, the spec and the event log at any point
curl -s $API/api/projects/$PROJECT/pipeline | python -m json.tool
curl -s $API/api/projects/$PROJECT/spec     | python -m json.tool
curl -s $API/api/projects/$PROJECT/events   | python -m json.tool
```

Built files land in `artifacts/<project-id>/dataset/` (`train.jsonl`,
`validation.jsonl`, `test.jsonl`, `dataset_card.json`), and each is registered as an
`Artifact` row with its size and SHA-256.

Edit the spec and re-run until it matches intent, then walk the pipeline with
`POST /api/projects/{id}/advance` (or `stages/{name}/run`) as later stages land.
A stage that is not built yet answers `501` with the milestone responsible, so
`advance` stops there instead of pretending.

**To build from your own data instead**, put a JSONL or CSV in the workspace and pass
the upload as the source with a column mapping:

```bash
curl -s -X POST $API/api/projects/$PROJECT/jobs -H 'Content-Type: application/json' -d '{
  "kind": "dataset_build",
  "params": {
    "build": true,
    "selected": ["upload:/absolute/path/tickets.jsonl"],
    "columns": {"text": ["subject", "body"], "label": "queue"},
    "target_examples": 500
  }
}'
```

Uploads are treated as untrusted: a `.jsonl`/`.csv` whitelist, a 64 MB size cap, a row
cap, and `errors="replace"` decoding.

### CLI reference

| Command | Purpose |
|---|---|
| `python -m slmforge serve [--reload]` | Run the FastAPI control plane (and the built SPA, if present) |
| `python -m slmforge worker` | Run the job worker — claims queued jobs, executes them in the sandbox |
| `python -m slmforge init-db` | Create the database schema |
| `python -m slmforge doctor` | Environment + capability report |
| `python scripts/demo_smoke.py` | Live end-to-end check (needs the API, a worker and network) |

---

## Configuration

All settings live in `config.py` and are read from the environment (or `.env`)
with the `SLMFORGE_` prefix. **Never read `os.environ` anywhere else.**

The settings most worth knowing:

| Variable | Default | Meaning |
|---|---|---|
| `SLMFORGE_LLM_PROVIDER` | `openrouter` | `openrouter` \| `ollama` \| `openai` \| `anthropic` \| `none` |
| `SLMFORGE_LLM_MODEL` | a free Nemotron model | Reasoning model used for the spec and the training plan |
| `SLMFORGE_LLM_BULK_MODEL` | a free Gemma model | Cheap model for high-volume synthetic generation |
| `SLMFORGE_LLM_API_KEY` | *(empty)* | Empty ⇒ the analyzer uses its offline heuristic and labels it as such |
| `SLMFORGE_SANDBOX_BACKEND` | `subprocess` | `docker` \| `subprocess` \| `hf_jobs` |
| `SLMFORGE_SANDBOX_NETWORK` | `deny` | `deny` \| `allowlist` \| `allow` |
| `SLMFORGE_MAX_VRAM_GB` | `8` | Model-ranking and plan budget |
| `SLMFORGE_DEPLOY_TARGET` | `local_openai` | `local_openai` \| `hf_endpoint` \| `hf_space` |
| `SLMFORGE_SEED` | `42` | Seed for every split and every sampling decision |
| `SLMFORGE_DATABASE_URL` | `sqlite:///./.data/slmforge.db` | Use `sqlite:///:memory:` in tests |

`.env.example` documents every knob. If you add a setting, document it there in
the same commit.

---

## Repository layout

```
src/slmforge/
  config.py            all settings (the only place env vars are read)
  db.py                engine/session factory + SQLite pragmas
  models.py            ORM: Project, StageRun, Job, Artifact, Deployment, Event
  log.py               logger factory; forces UTF-8 stdio on legacy consoles
  net.py               retry/backoff for every outbound call
  schemas/             the pydantic contracts exchanged between stages
    spec.py              MLSpec — the central document of the product
    dataset.py           DatasetPlan, DatasetCard, candidate/label/quality types
    model.py             ModelCandidate, ModelSelection, FineTuneSupport
    training.py          TrainingPlan, TrainingRecord, Hyperparameters
    evaluation.py        EvalRun, EvalReport, Comparison
    deployment.py        DeployRequest, DeployRecord, UsageStats
    common.py            TaskType, OutputFormat, Provenance, enums
  pipeline/
    base.py              Stage / StageContext / StageResult abstraction
    analyzer.py          Stage 1 — prompt → MLSpec  (implemented)
    datasets.py          Stage 2 — discovery, curation, splits
    models.py            Stage 3 — candidate ranking
    baseline.py          Stage 4 — pre-training measurement
    planner.py           Stage 5 — fine-tune vs prompt/RAG, method, estimate
    trainer.py           Stage 6 — generate + run the training script
    evaluator.py         Stage 7 — baseline vs adapted comparison
    deployer.py          Stage 8 — publish and serve an endpoint
    orchestrator.py      persistence, events, status transitions, approval gates
  sandbox/
    base.py              Sandbox interface + SandboxResult (stdout/stderr/log are distinct)
    subprocess_backend.py  dev fast path; process isolation, not a security boundary
    docker_backend.py    the real boundary: network/fs/user/memory limits
  jobs/
    queue.py             durable queue with leases
    worker.py            the claim/execute/complete loop
    handlers.py          job kind → stage/function mapping
  integrations/
    hf_cli.py            primary Hub interface (models, datasets, parquet, Jobs)
    parquet.py           DuckDB profiling + single-shot local materialisation
    huggingface.py       discovery, licence normalisation, publishing
    ollama.py            local model inventory; degrades to [] when offline
    llm.py               OpenAI-compatible planner client + JSON repair
  api/
    routes/              projects, stages, jobs, deployments, health
    payloads.py          HTTP request/response models
    deps.py              session/project dependencies
  main.py              app factory; serves the API and the built SPA
  cli.py               serve | worker | init-db | doctor
web/                   Vite + React + TypeScript dashboard
docker/sandbox.Dockerfile   the pinned, unprivileged sandbox image
scripts/demo_smoke.py  live end-to-end check against the real API, worker and Hub
tests/                 218 tests; external services are always mocked
docs/                  ARCHITECTURE.md, ROADMAP.md
```

---

## Architecture

```
Browser (Vite + React)  ──/api──▶  FastAPI control plane
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
              Orchestrator       Job queue (SQLite)   Project/artifact DB
                    │                 │
                    ▼                 ▼
            Pipeline stages  ◀──  Worker process
                    │
                    ▼
            Sandbox (subprocess → Docker)
                    │
        ┌───────────┼────────────┬──────────────┐
        ▼           ▼            ▼              ▼
   HF datasets   Ollama /     PEFT + TRL    HF Hub / local
   + uploads     HF models    (LoRA/QLoRA)  OpenAI-compatible server
```

Two invariants shape every design decision:

1. **The API process never does heavy work.** Anything that can take minutes is a
   job kind, claimed by a worker, executed in a sandbox.
2. **Stages never touch the database.** A stage receives a `StageContext` and
   returns a `StageResult`; the `Orchestrator` is the only thing that writes
   state, emits events and applies the status transition.

Full detail, including the sandbox/security model: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Testing and quality gates

Run all four before you push; CI runs the same commands.

```bash
.venv/Scripts/python -m pytest                     # full suite
.venv/Scripts/python -m ruff check src tests       # lint
.venv/Scripts/python -m ruff format src tests      # format
.venv/Scripts/python -m mypy src                   # types
```

House rules for tests (they are not negotiable, they are how the product stays honest):

- **Contract tests, not internals.** Test the promise a stage makes, not its helpers.
- **Never depend on the network, a GPU, or a running Ollama.** Mock with `respx`
  or monkeypatch; `/api/capabilities` and `doctor` are how you check reality.
- **DB tests use in-memory SQLite**; API tests override the session dependency.
- **Assert on `SandboxResult.stdout`**, never on the log file — the log header
  echoes the command source, so log assertions can pass for the wrong reason.
- **Tiny data, separable tasks.** A test that needs a GPU or ten minutes is a test
  nobody runs.
- **A test that passes because it asserts nothing is a bug.**

`tests/test_sandbox_docker.py` is the one exception: it needs a live Docker daemon
and **skips when it is down** — but it is also the suite that proves the security
posture (network denial, read-only root, non-root user, memory ceiling, timeout
cleanup). Run it with Docker started before touching `sandbox/`.

### The live smoke test

Unit tests mock the world, so one thing they cannot tell you is whether the demo still
works. [`scripts/demo_smoke.py`](scripts/demo_smoke.py) does: it drives the real API and
a real worker through analyze → plan → approved build, then reconciles the delivered
JSONL against the `DatasetCard` (row counts, SHA-256, chat format) and the project's
registered artifacts.

```bash
.venv/Scripts/python -m slmforge serve     # terminal 1
.venv/Scripts/python -m slmforge worker    # terminal 2
.venv/Scripts/python scripts/demo_smoke.py # terminal 3 — needs network + worker
```

It exits non-zero if any check fails, and it prints every finding (licence, missing
class, removed leaks) so a reviewer can see what the pipeline actually claimed.

---

## Contributing

### The workflow

1. **Read [`docs/ROADMAP.md`](docs/ROADMAP.md) first.** It is the agenda, and the
   next task is normally "the next unstarted stage" or the next unmet exit
   criterion of the current one.
2. **Branch per concern.** `git checkout -b stage-2-dataset-curation`. Keep one
   concern per commit; imperative subject line.
3. **Implement the stage.** Read its docstring in `pipeline/<stage>.py` — that
   docstring *is* the spec. Keep the algorithm and the docstring true to each other.
4. **Wire the long part as a job.** If a step can take minutes, it belongs in
   `jobs/handlers.py`, not in a request handler.
5. **Flip `implemented = True`** on the stage class and delete its `NotImplementedError`.
   `GET /api/projects/{id}/pipeline` reads that flag, so the UI unlocks honestly.
6. **Update the books in the same commit**: the stage docstring,
   `docs/ROADMAP.md` status + exit criteria checkboxes, `README.md` if
   user-facing behaviour changed, and `.env.example` for any new setting.
7. **Verify for real.** `pytest`, `ruff`, `mypy`, plus one *live* run of the stage
   through the API or CLI. A green test suite is necessary, not sufficient — the
   stage is done when its roadmap **exit criteria pass on a real run**.

### Conventions

- Python 3.11+, `from __future__ import annotations` at the top of every module,
  full type hints, line length 100, `ruff` rules enforced.
- Pydantic models in `schemas/` are the public vocabulary. **Add fields with
  defaults; never repurpose an existing one.** Store enums as `str, enum.Enum`.
- Return `StageResult(status="failed", error=...)` for failures — do not raise out
  of `run()`. A clear error beats a stack trace.
- Logging via `get_logger("area")` from `slmforge.log`. No bare `print` in library
  code; CLI output goes through `rich.console`.
- HTTP routers stay thin: validate, delegate, return.
- **Comments explain *why***, especially for heuristics (why QLoRA here, why this
  threshold). The *what* is the code's job.
- Sign commits with:

  ```
  Generated with Codebuff 🤖
  Co-Authored-By: Codebuff <noreply@codebuff.com>
  ```

### Guardrails — do not weaken these

1. **Evidence over enthusiasm.** Never report success because a job exited 0.
   Success is "the adapted model beat the baseline on held-out data", or an
   honest "don't fine-tune".
2. **Transparency and consent.** Training and deployment need an approval gate, a
   pre-flight estimate, and a plain-English record of what was chosen and why.
3. **Untrusted input stays untrusted.** Uploaded data and *generated training
   scripts* are never executed in the API or worker process — they go through the
   `Sandbox`, and the subprocess backend must be described as what it is.
4. **Secrets never land anywhere durable.** `HF_TOKEN` and API keys live in the
   environment; they never appear in the repo, logs, artifacts or the database.
   `Deployment.auth_ref` stores a *reference*, never a key.
5. **Provenance always.** Every dataset and model records origin, licence and
   commercial-use verdict. Synthetic examples are labelled and never counted as
   ground truth.
6. **No leakage.** Splits are seeded and checked; test data never enters training.
7. **High-stakes work is flagged**, never silently treated as an ordinary build.

If a change makes SLMForge *look* better while reporting less truthfully, it is wrong.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker_available()` false / stage says Docker is down | Daemon stopped | Start Docker Desktop, or set `SLMFORGE_SANDBOX_BACKEND=subprocess` and accept the weaker isolation |
| A queued job sits at `running` forever | A stale worker from an earlier session, or a handler completing through a second session | On Windows `pkill` does **not** kill native processes — use `taskkill //PID <pid> //F` or `Get-CimInstance Win32_Process` and check for duplicate `slmforge worker` processes |
| A job never leaves `pending` | No worker running | Start `python -m slmforge worker` (it prints `Worker <id> started`) |
| `Expected JSON from \`hf datasets parquet …\`` | The Hub CLI prints a **table** for that subcommand unless `--format json` is passed | Already handled by `hf_cli.run_json`; if you add a new `hf` call, add a test rather than guessing which subcommands need the flag |
| `hf datasets info` returns `card_data`, not `cardData` | The CLI emits snake_case for `info` commands | Read fields defensively; use `hf_cli.dataset_license()` rather than assuming a shape |
| `hf` CLI "not found on PATH" | Installed into `.venv` instead of globally | `pip install -U "huggingface_hub[cli]"` on the **global** interpreter |
| `DuckDBUnavailable` during dataset profiling | `duckdb` missing where `hf` can see it | `pip install duckdb` globally |
| `SSL connect error` / `WinError 10054` | Reference network resets TLS with `huggingface.co` under load | Expected. Everything routes through `net.retry_call`; prefer one remote profile then one local materialisation |
| `UnicodeEncodeError` writing to the console | Windows legacy code page | `log.force_utf8_stdio()` handles the streams; keep CLI *output* ASCII anyway |
| `http://127.0.0.1:5173` refused | Vite bound IPv6 only | Use `http://localhost:5173`, or `npm run build` and serve from FastAPI |
| `torch.cuda.is_available()` is `False` on a machine with a GPU | CPU-only torch wheel | Install a CUDA wheel, or accept CPU-only (fine for the demo's baseline) |
| Training "succeeds" instantly with no adapter | Non-zero exit was treated as success somewhere | That is a bug — check `SandboxResult.exit_code` and report it |
| Model cache filling the disk | `HF_HOME` outside the workspace | Keep `SLMFORGE_HF_HOME` under `./.cache/` (gitignored) and use `allow_patterns` |

---

## Scope and non-goals

**In scope (the demo):** one happy path — ticket classification into four labels —
taken end to end on a single machine, with real numbers and honest verdicts.

**Explicitly out of scope:** multi-user accounts and auth on the SLMForge API,
horizontal scaling, a scheduler with priorities and per-GPU concurrency control,
object storage, Alembic migrations, and any claim of production readiness. These
are listed as limitations in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#8-known-limitations-honest-list)
rather than quietly assumed.

**No paid Hugging Face plan is required.** This account has no credits, so
`hf jobs` (GPU training) and `hf endpoints` (hosting) are implemented-but-gated
paths, not defaults. The demo trains locally and serves locally via an
OpenAI-compatible server; free GPU alternatives (local CUDA torch, Kaggle, or
Lightning AI) are evaluated in [docs/ROADMAP.md](docs/ROADMAP.md).

### Definition of done for the prototype

The MVP is demonstrable when, from one sentence, a user can:

1. get a specification they can edit,
2. see the datasets and the model that were chosen, **and why**,
3. read a baseline,
4. approve and watch a fine-tune run inside the sandbox,
5. see an honest before/after comparison — including "it did not help" when true,
6. click deploy and call a hosted endpoint.

---

## Licence

Apache-2.0.
