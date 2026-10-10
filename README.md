# SLMForge

**Turn a natural-language requirement into a task-specific Small Language Model.**

SLMForge is a prompt-to-SLM platform. You describe the model you need in plain
language; SLMForge interprets the request, curates a dataset from the purpose,
finds candidate small models on **Ollama** and the **Hugging Face Hub**, runs
them in an isolated sandbox, decides whether fine-tuning is even justified,
picks the optimal adaptation method, generates and runs the training script, and
hands back a **callable endpoint** — a locally hosted OpenAI-compatible server by
default, or a Hugging Face Inference Endpoint / Space once you have credits.
Datasets, models, training Jobs and deployments all go through the **official `hf`
CLI**, which carries its own authentication.

> **Status: early prototype (Stage 0 — foundation).** The architecture, contracts,
> API skeleton, job queue, sandbox interface and docs are in place. The eight
> pipeline stages are stubbed with their intended algorithms documented; they are
> implemented milestone by milestone. See [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Why

Building a specialised model today means stitching together dataset hunting,
cleaning, model comparison, licence checks, GPU plumbing, training libraries,
evaluation and serving. A developer who knows *exactly* what they want the model
to do can still spend days on everything except the actual task.

SLMForge automates that lifecycle **without hiding the decisions that matter**:
you review the dataset, the model, the training plan, the cost estimate and the
before/after evaluation before you commit to anything expensive.

**Core workflow**

```
Describe → Specify → Prepare Data → Select Model → Baseline
        → Plan → Fine-tune → Evaluate → Deploy → Hosted endpoint
```

SLMForge is explicitly *not* a fine-tune-everything button. If a good prompt, a
RAG setup, or a different base model is the better answer, it says so — and a
build that correctly ends in "don't fine-tune" is a success, not a failure.

---

## How it works

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Analyze** | Prompt → validated, user-editable `MLSpec` (task, I/O contract, metrics, constraints, risks). |
| 2 | **Curate data** | Search approved dataset sources, validate, dedupe, map labels, split, track provenance. |
| 3 | **Select model** | Rank candidates from Ollama (local) + HF Hub by task fit, size, licence and VRAM. |
| 4 | **Baseline** | Measure candidates *before* training so improvement can be proven. |
| 5 | **Plan** | Choose fine-tune vs. prompt-only vs. RAG; pick SFT/LoRA/QLoRA; estimate cost & time. |
| 6 | **Train** | Generate a reproducible script, run it in the sandbox, stream logs, checkpoint. |
| 7 | **Evaluate** | Compare baseline vs. adapted on held-out data; report regressions honestly. |
| 8 | **Deploy** | Publish to the HF Hub and return an authenticated endpoint with usage/limits. |

Each stage has a typed contract in `src/slmforge/schemas/`, its algorithm
documented in `src/slmforge/pipeline/`, and an approval gate where money or
irreversibility is involved.

---

## Architecture at a glance

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
   HF datasets   Ollama /     PEFT + TRL    HF Hub / HF Inference
   + uploads     HF models    (LoRA/QLoRA)  Endpoints (deploy)
```

The API is a **control plane**: it never trains in-process. All heavy work is a
job, executed by a worker inside a sandbox with CPU/memory/time limits.

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick start

Requirements: Python ≥ 3.11 and `uv`; Node ≥ 20 for the web UI.

Before you start:

| | Needed for | Why |
|---|---|---|
| `hf` CLI + `hf auth login` | models, datasets, Jobs, endpoints | Our primary Hub interface; it holds the token so the app doesn't |
| `duckdb` | dataset profiling (`pip install duckdb`) | Also required by `hf datasets sql` (on the **global** interpreter) |
| OpenRouter key (`OPENROUTER_API_KEY`) | parsing prompts, choosing the method | The planner LLM. Free models are supported and are the default |
| An HF OAuth app (`HF_CLIENT_ID` + `HF_CLIENT_SECRET`) | “Sign in with Hugging Face” | Optional — the dashboard runs anonymously without it |
| Docker Desktop running | the `docker` sandbox | Real isolation for untrusted work (dev can use `subprocess`) |
| A CUDA build of torch | local fine-tuning | The stock build here is CPU-only, so training is impractical without it |

```bash
# 1. Environment (inherits the machine's ML stack if present)
uv venv --system-site-packages .venv
uv pip install -e . -p .venv/Scripts/python.exe      # Windows
# uv pip install -e . -p .venv/bin/python             # macOS / Linux

# 2. Configuration
cp .env.example .env        # then edit as needed

# 3. Check what this machine can actually do
.venv/Scripts/python -m slmforge doctor

# 4. Build the sandbox image (the default `docker` backend needs it)
docker build -f docker/sandbox.Dockerfile -t slmforge/sandbox:latest .
# Not running Docker? Set SLMFORGE_SANDBOX_BACKEND=subprocess in .env — but note
# that mode is process isolation, not a security boundary.

# 5. Create the database and start the API
.venv/Scripts/python -m slmforge init-db
.venv/Scripts/python -m slmforge serve --reload
```

API docs: <http://127.0.0.1:8000/docs> · capabilities probe:
<http://127.0.0.1:8000/api/capabilities>

In a second terminal, start a worker (needed for training/evaluation jobs):

```bash
.venv/Scripts/python -m slmforge worker
```

Web UI (optional, in development):

```bash
cd web && npm install && npm run dev     # http://localhost:5173, proxies /api
```

> On some machines Vite binds IPv6 only, so `http://127.0.0.1:5173` is refused while
> `http://localhost:5173` works. Use `localhost`. Alternatively run `npm run build`
> and let FastAPI serve the built UI at <http://127.0.0.1:8000> — no second server.

### Try the API

```bash
curl -s -X POST http://127.0.0.1:8000/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Classify support tickets into Billing, Technical, Account Access or General and return JSON."}'

curl -s http://127.0.0.1:8000/api/projects/<id>/pipeline
curl -s -X POST http://127.0.0.1:8000/api/projects/<id>/stages/analyze/run \
  -H 'Content-Type: application/json' -d '{}'
```

---

## Repository layout

```
src/slmforge/
  config.py        validated settings (env-driven)
  db.py            engine/session + SQLite pragmas
  models.py        ORM: Project, StageRun, Job, Artifact, Deployment, Event
  schemas/         stage contracts (MLSpec, DatasetPlan, TrainingPlan, EvalReport, ...)
  pipeline/        the eight stages + orchestrator (transparency & approval gates)
  sandbox/         isolated execution: subprocess now, Docker next
  jobs/            durable queue, worker loop, job handlers
  integrations/    Ollama, Hugging Face Hub, planner LLM (OpenAI-compatible)
  api/             FastAPI routers + HTTP payloads
  cli.py           serve | worker | init-db | doctor
web/               Vite + React + TypeScript dashboard
tests/             pytest suite
docs/              ARCHITECTURE.md, ROADMAP.md
```

---

## Design principles

1. **Prompt-first** — describe outcomes, not training commands.
2. **Transparency** — automate the work, expose the decisions and the evidence.
3. **Evaluate before training** — a baseline first; prove the gain or report none.
4. **Provenance** — every dataset and model records its origin and licence.
5. **Reproducibility** — versions, configs, seeds and script hashes are stored.
6. **Resource awareness** — estimate VRAM/time/cost *before* an expensive run.
7. **Safe execution** — untrusted data and scripts run out of process, under limits.
8. **User approval** — no costly training or deployment without a click.
9. **Honest reporting** — uncertainty, limitations and negative results are surfaced.
10. **Portability** — artifacts stay exportable; no provider lock-in by default.

---

## Safety and responsible use

Uploaded datasets and generated scripts are treated as **untrusted input**. Jobs
run under CPU/memory/time limits with a scrubbed environment and network policy.
High-stakes requests (medical, legal, financial, employment) are flagged by the
analyzer and require additional safeguards — completing training never implies a
model is safe for such use. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#5-sandbox-and-security-model).

## Licence

Apache-2.0.
