# SLMForge Architecture

This document explains how SLMForge is put together: the components, the data
that flows between them, the execution model for long jobs, and the security
model for untrusted input. It is written to be true of the code as it stands —
where something is not built yet, it says so.

See also: [ROADMAP.md](ROADMAP.md) (the staged agenda), [AGENTS.md](../AGENTS.md)
(conventions and guardrails).

---

## 1. Shape of the system

SLMForge is a **modular monolith** with a separate worker process. The design
brief calls for a logical architecture of eight components; a prototype that
splits those into eight microservices would be slower to build and harder to
reason about, so they map onto modules instead.

```
┌──────────────────────────────────────────────────────────────┐
│  Browser — Vite + React dashboard (web/)                     │
│  capabilities · prompt form · pipeline · spec · logs         │
└───────────────────────────┬──────────────────────────────────┘
                            │ /api (JSON)
┌───────────────────────────▼──────────────────────────────────┐
│  FastAPI control plane (src/slmforge/api, main.py)            │
│  validate → delegate → respond.  Never trains in-process.     │
└───────┬──────────────────────────────┬───────────────────────┘
        │                              │
┌───────▼──────────────┐   ┌───────────▼──────────────────────┐
│  Orchestrator        │   │  Job queue (SQLite table + lease)│
│  StageRun/Event rows │   │  jobs/queue.py                   │
│  status transitions  │   └───────────┬──────────────────────┘
└───────┬──────────────┘               │ claim
        │                              │
┌───────▼──────────────────────────────▼───────────────────────┐
│  Pipeline stages (pipeline/)                                 │
│  analyze · curate_data · select_model · baseline ·           │
│  plan_training · train · evaluate · deploy                   │
└───────┬──────────────────────────────────────────────────────┘
        │ run untrusted work
┌───────▼──────────────────────────────────────────────────────┐
│  Sandbox (sandbox/)  subprocess now → Docker next            │
│  work dir · scrubbed env · rlimits · timeout · network policy │
└───────┬──────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│  External: HF Hub (datasets, models, publish) · Ollama ·      │
│  planner LLM (OpenAI-compatible) · HF Inference Endpoints     │
└──────────────────────────────────────────────────────────────┘
```

### Component responsibility map

The eight functional modules from the product brief map as follows.

| Brief module | Code |
|---|---|
| 6.1 Prompt and Task Analyzer | `pipeline/analyzer.py` + `integrations/llm.py` |
| 6.2 Dataset Discovery and Preparation | `pipeline/datasets.py` + `integrations/huggingface.py` |
| 6.3 Model Discovery and Compatibility | `pipeline/models.py` + `integrations/ollama.py`, `huggingface.py` |
| 6.4 Training Planner | `pipeline/planner.py` |
| 6.5 Sandboxed Execution Engine | `sandbox/` + `jobs/` |
| 6.6 Evaluation and Model Selection | `pipeline/baseline.py`, `pipeline/evaluator.py` |
| 6.7 Deployment and Endpoint Management | `pipeline/deployer.py` + `api/routes/deployments.py` |
| 6.8 Project Dashboard | `web/` + `api/routes/projects.py` (events, artifacts) |

---

## 2. Data flow

### 2.1 The synchronous path (fast, no compute)

```
POST /api/projects            → Project row (status=draft) + Event
POST /api/projects/{id}/stages/analyze/run
                              → Orchestrator → AnalyzerStage → MLSpec
                              → project.spec, status=specified, Event
PUT  /api/projects/{id}/spec  → user edits persisted before anything is spent
```

### 2.2 The asynchronous path (expensive work)

```
POST …/advance or a UI action
   → JobQueue.enqueue(kind)            status=pending
Worker.run_once()
   → JobQueue.claim_next()             status=running, lease taken
   → handler → Orchestrator.run_stage()
       → Stage.run_timed()             → Sandbox.run(command)
                                        → log file, artifacts on disk
   → JobQueue.complete()               status=succeeded | failed
```

Every stage execution and every queue transition writes an `Event`, which is what
the dashboard's build log renders. Nothing important happens silently.

### 2.3 Persistence

Six tables (`src/slmforge/models.py`):

| Table | Purpose |
|---|---|
| `projects` | The build: prompt, status, and the JSON of every approved artifact (spec, dataset plan, model choice, training plan, eval report). |
| `stage_runs` | One row per stage execution: status, attempt, timings, output, error. |
| `jobs` | Durable queue entries with claim/lease bookkeeping and usage. |
| `artifacts` | Files produced (dataset splits, checkpoints, reports, logs) plus provenance metadata. |
| `deployments` | Published endpoints, revisions, usage, and an auth **reference**. |
| `events` | Append-only audit/telemetry log for the dashboard. |

SQLite is the default (`SLMFORGE_DATABASE_URL`) with WAL mode, foreign keys on,
and a busy timeout so the API and worker can share one file. The schema is created
with `create_all`; Alembic migrations replace that once the schema stabilises.

**Why JSON columns for stage output:** the contracts are versioned Python models
(`schemas/`), and storing their JSON keeps the schema stable while the contracts
evolve. Every stage output is validated by pydantic before it is stored.

---

## 3. Contracts between stages

The `schemas/` package is the vocabulary of the system. Each stage consumes one
contract and produces the next.

```
prompt ──▶ MLSpec ──▶ DatasetPlan ──▶ (DatasetCard) ──▶ ModelSelection
                                                            │
              EvalReport ◀── TrainingRecord ◀── TrainingPlan ◀
                  │
                  ▼
             DeployRecord
```

Key contracts:

- **`MLSpec`** — task type, input/output contract, allowed labels, evaluation plan,
  constraints, deployment preference, plus honesty fields: `assumptions`,
  `open_questions`, `out_of_scope`, `risk_notes`, `is_high_stakes`, `readiness`.
- **`DatasetPlan` / `DatasetCard`** — the proposed data strategy, and what was
  actually built: per-source `Provenance` (URI, revision, licence, transformations,
  synthetic flag), quality issues, split sizes, file hashes.
- **`ModelCandidate` / `ModelSelection`** — parameters, context window, licence with
  `commercial_ok`, `FineTuneSupport` (methods, min VRAM), a `rank_score` that must
  come with a `rank_rationale`.
- **`TrainingPlan`** — `recommendation` (finetune / prompt_only / rag / no_finetune),
  `method` (sft / lora / qlora), full `Hyperparameters`, a `ResourceEstimate`, and
  the `config_fingerprint` of the generated script.
- **`EvalRun` + `Comparison` + `EvalReport`** — per-split metrics, valid-output rate,
  latency percentiles, confusion matrix, error samples, and an explicit `verdict`
  (`accept_finetuned` / `keep_baseline` / `inconclusive` / `needs_more_data`).
- **`DeployRequest` / `DeployRecord`** — what to publish, then what exists: endpoint,
  `auth_ref` (a reference, never a key), request/response examples, known limitations.

Contract rules:

- Additive changes only (new fields with defaults) while the prototype is young.
- Enums are `StrEnum` so they serialise as plain strings into JSON columns.
- A `ResourceEstimate` is populated *before* any costly stage, so the user sees
  VRAM/time/cost up front rather than after the fact.

---

## 4. Execution model

### 4.1 Stages

`Stage` (`pipeline/base.py`) is an abstract class with one method:

```python
def run(self, ctx: StageContext) -> StageResult
```

`StageContext` carries the project, the prompt, the artifact and work directories,
settings, per-stage `params`, and `upstream` results from earlier stages. It
deliberately contains **no database session**.

`Stage.run_timed()` converts an exception into
`StageResult(status="failed", error="TypeError: ...")` so a crash is recorded as a
failed stage, not a lost request. `Stage.implemented` is `False` until the stage's
milestone lands, and the API reports that flag so the UI can be honest instead of
failing at click time.

Two flags shape the user experience:

- `requires_approval` — the orchestrator/API refuses to run the stage without an
  explicit `approved=true` (`HTTP 428` otherwise).
- Long stages (`train`, `baseline`, `evaluate`) cannot be called synchronously at
  all — the API returns `409` and they must go through the queue, so an HTTP
  request can never hang for an hour.

### 4.2 Orchestrator

`pipeline/orchestrator.py` is the only module that mutates project state for the
pipeline. For each run it: creates a `StageRun`, emits a start event, builds the
context, executes the stage, records status/output/error/duration, emits a result
event, and — on success — copies the output onto the project row and advances the
project status (`_STAGE_TO_COLUMN`, `_STAGE_TO_STATUS`).

### 4.3 Queue and worker

`jobs/queue.py` is a SQLite-backed queue with a lease:

- `enqueue` → `pending`
- `claim_next` → `running`, stamping `claimed_by` / `claimed_at` / `attempts`
- `heartbeat` extends the lease; a job whose lease expires (15 min) is reclaimed
  by `_reap_expired`, so a crashed worker does not strand work
- `complete` records status, exit code, usage, logs
- `requeue` retries until `max_attempts`, then fails

`jobs/worker.py` polls, dispatches to a handler by `JobKind`, and is written so a
throwing handler can never kill the loop. `run_once()` exists to make the worker
testable without threads or sleeps.

### 4.4 Idempotency and cancellation

- A stage that already succeeded is skipped by `/advance` unless `force=true`.
- `JobQueue.request_cancel` cancels pending jobs; cancelling a *running* job is the
  worker's job (SIGTERM then SIGKILL, Stage 6 work).
- Deployment and training are never automatic: both sit behind approval gates.

---

## 5. Sandbox and security model

### 5.1 What is untrusted

Uploaded datasets, HF dataset contents, and — most importantly — **generated
training scripts** are untrusted. They are never imported into or executed by the
API or worker process.

### 5.2 Current implementation: `SubprocessSandbox`

`Sandbox.run(command, work_dir, limits, env, log_path)`:

- runs in a per-job work directory under the project's artifact root;
- scrubs the environment to an allow-list (`PATH`, `SYSTEMROOT`, `HOME`, `TMPDIR`,
  `TEMP`) and sets `HF_HOME` inside the workspace. **`HF_TOKEN` and other secrets
  are not forwarded**; a job that needs one must be given a scoped value explicitly
  via the `env` argument;
- applies a wall-clock timeout, and on POSIX, `RLIMIT_CPU` and `RLIMIT_AS` via a
  `preexec_fn` (Windows has no equivalent here, so the timeout is the only bound);
- starts a new process session and terminates the whole process group on timeout;
- captures a combinational log file and lists produced files.

> **Be precise about what this is:** process isolation and resource limiting, *not*
> a security boundary. A malicious script in this sandbox can still read the
> filesystem and reach the network. Docker is the intended real boundary
> (`SLMFORGE_SANDBOX_BACKEND=docker`, Stage 4), and it fails loudly today rather
> than silently downgrading to the weaker mode.

### 5.3 Secret handling

- Configuration comes from the environment; the repo contains no secrets.
- `Deployment.auth_ref` stores a *reference* to an API key, never the key.
- `.gitignore` excludes `.env`, `*.pem`, `*.key`, `.data/`, `artifacts/`.
- Artifacts and logs are local files under `SLMFORGE_ARTIFACT_ROOT`.

### 5.4 Data protection

- Splits are made with a fixed seed and checked for cross-split near-duplicates, so
  evaluation numbers are not inflated by leakage.
- Dataset provenance (source, licence, transformations, synthetic flag) is recorded
  per contributing source; synthetic examples are labelled and never counted as
  ground truth.
- High-stakes domains are flagged in `MLSpec.is_high_stakes` and never silently
  treated as an ordinary build.
- Deletion controls for uploads/artifacts are a Stage 9 deliverable and are called
  out as missing rather than implied.

---

## 6. Integrations

- **Planner LLM** (`integrations/llm.py`) — any OpenAI-compatible chat endpoint,
  including a local Ollama server at `/v1`. Used to interpret the prompt and plan.
  If none is configured or reachable, `PlannerLLMUnavailable` is raised and the
  stage must fall back to a deterministic heuristic instead of pretending to reason.
- **Ollama** (`integrations/ollama.py`) — local model inventory (`/api/tags`), plus
  a fully local, zero-cost planner option. Degrades gracefully: an unreachable
  server yields `[]`, never an exception.
- **Hugging Face** (`integrations/huggingface.py`) — discovery (`list_models`,
  `list_datasets`), licence normalisation for commercial-use risk
  (`is_commercial_friendly`), dataset download with `allow_patterns`, and publishing
  (`create_repo`, `upload_folder`, model card). Imported lazily so the API runs
  without the ML extra.

---

## 7. Observability

- **Events** — every project has an append-only event stream with level, stage,
  message and structured data; the dashboard renders it as the build log.
- **StageRun rows** — attempt count, duration, output, error, for retry history.
- **Job logs** — the sandbox writes `<job>.log` into the work directory and the path
  is recorded on the job.
- **`/api/capabilities`** — reports GPU/torch/Ollama/planner/HF state with warnings,
  so environment problems surface before a build rather than mid-training.
- **`slmforge doctor`** — the same checks for the terminal.

---

## 8. Known limitations (honest list)

1. All eight pipeline stages are documented stubs; `run()` raises
   `NotImplementedError` with its milestone.
2. `torch` on the reference machine is a CPU-only build, so training there is
   impractical until a CUDA wheel is installed.
3. The subprocess sandbox is not a security boundary; Docker is not implemented.
4. No Alembic migrations yet — schema changes mean recreating the SQLite file.
5. Single-worker model: no scheduler, no priority, no per-GPU concurrency control.
6. Artifact storage is a local directory; no retention policy or object storage yet.
7. Endpoint lifecycle (stop/delete) returns `501` until Stage 8.
8. No authentication on the API — it binds loopback and is single-user by
   assumption. Multi-user isolation is out of MVP scope.
