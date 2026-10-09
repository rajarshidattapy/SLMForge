# AGENTS.md — working in the SLMForge repository

Guidance for AI coding agents (and humans) contributing to SLMForge. Read this
before changing code; it covers what the product is, how the repo is laid out,
the rules that keep the pipeline honest, and the environment quirks that bite.

---

## 1. What this project is

SLMForge turns a natural-language requirement into a task-specific Small
Language Model: **Describe → Specify → Prepare Data → Select Model → Baseline →
Plan → Fine-tune → Evaluate → Deploy**.

Two rules define the product and must never be violated:

1. **Evidence over enthusiasm.** Never report success because a job exited 0.
   Success means the adapted model beat the baseline on held-out data, or the
   system correctly said "don't fine-tune".
2. **Transparency and consent.** Every costly or irreversible action (training,
   deployment) needs an explicit approval gate, a pre-flight estimate, and a
   plain-English record of what was chosen and why.

If a change makes SLMForge look better but report less truthfully, it is wrong.

---

## 2. Repository map

| Path | Responsibility |
|------|----------------|
| `src/slmforge/config.py` | All settings. **Never read `os.environ` elsewhere.** |
| `src/slmforge/models.py` | ORM tables: Project, StageRun, Job, Artifact, Deployment, Event. |
| `src/slmforge/schemas/` | Pydantic contracts exchanged between stages. The public vocabulary. |
| `src/slmforge/pipeline/` | The eight stages (`base.py` = abstraction, `orchestrator.py` = persistence/events). |
| `src/slmforge/sandbox/` | Isolated execution. `base.py` = interface, `subprocess_backend.py` = current impl. |
| `src/slmforge/jobs/` | `queue.py` (durable queue), `worker.py` (loop), `handlers.py` (job → stage). |
| `src/slmforge/integrations/` | `ollama.py`, `huggingface.py`, `llm.py` (planner LLM). |
| `src/slmforge/api/` | FastAPI routers, dependencies, HTTP payloads. |
| `src/slmforge/main.py` | App factory; serves the API and the built SPA. |
| `web/` | Vite + React + TypeScript dashboard. |
| `docs/ROADMAP.md` | The staged development agenda — **check it before starting work.** |
| `docs/ARCHITECTURE.md` | Data flow, contracts, sandbox/security model. |

---

## 3. Commands

```bash
# Setup (Windows/Git Bash shown; use .venv/bin/python on POSIX)
uv venv --system-site-packages .venv
uv pip install -e . -p .venv/Scripts/python.exe

# Run
.venv/Scripts/python -m slmforge doctor
.venv/Scripts/python -m slmforge init-db
.venv/Scripts/python -m slmforge serve --reload
.venv/Scripts/python -m slmforge worker

# Quality (install the dev extra first)
uv pip install -e '.[dev]' -p .venv/Scripts/python.exe
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests
.venv/Scripts/python -m ruff format src tests
.venv/Scripts/python -m mypy src

# Web
cd web && npm install && npm run dev     # :5173, proxies /api → :8000
```

Prefer `uv pip install -e .` over ad-hoc `pip install` so the lockfile and
metadata stay consistent.

---

## 4. Conventions

- **Python:** 3.11+, `from __future__ import annotations` at the top of every
  module, full type hints, `ruff` badge rules. Line length 100.
- **Pydantic:** the `schemas/` models are the API of the pipeline. Changes there
  ripple everywhere — add fields with defaults, don't repurpose existing ones.
  Store enums as `str, enum.Enum` so they serialize cleanly into JSON columns.
- **Stages:** a stage must be pure with respect to the database. It receives a
  `StageContext` and returns a `StageResult`; the `Orchestrator` persists state,
  emits events and applies the project status transition in `_STAGE_TO_STATUS`.
- **Errors:** a stage failure becomes `StageResult(status="failed", error=...)` —
  don't raise out of `run()`; `Stage.run_timed` already converts exceptions, but
  returning a clear error is better than a stack trace.
- **Logging:** `get_logger("area")` from `slmforge.log`. No bare `print` in
  library code (CLI output uses `rich.console`).
- **HTTP:** routers stay thin — validate, delegate, return. Business logic lives
  in `pipeline/` or `jobs/`.
- **Long work never runs in the API process.** If something can take minutes, it
  is a job kind handled by the worker through the sandbox.
- **Comments** explain *why*, especially for heuristics (e.g. why QLoRA here).
  Docstrings on stage classes describe the intended algorithm — keep them true.
- **Commits:** imperative subject, one concern each, and end with:

  ```
  Generated with Codebuff 🤖
  Co-Authored-By: Codebuff <noreply@codebuff.com>
  ```

---

## 5. Guardrails (do not weaken these)

- Untrusted input: datasets, uploads and **generated training scripts** are never
  executed in the API/worker process. They go through `Sandbox`.
- The subprocess sandbox is **process isolation, not a security boundary** — say
  so; don't claim sandboxing it doesn't provide. Docker is the real boundary.
- Secrets (`HF_TOKEN`, API keys) live in the environment/secret store, never in
  the repo, logs, artifacts or the database. `Deployment.auth_ref` stores a
  *reference*, never a key.
- Every dataset and model records provenance and licence; synthetic examples are
  labelled and never counted as ground truth.
- Never leak test data into training data; splits are made with the configured seed.
- Don't auto-deploy or auto-train. Approval gates are product requirements.
- High-stakes domains (medical/legal/financial/employment) get `is_high_stakes`
  and the corresponding safeguards — never silently treated as a normal build.

---

## 6. Environment notes (reference dev machine)

- **Windows + Git Bash.** Use POSIX shell syntax (`mv`, `rm`, `/dev/null`). Use
  `.venv/Scripts/python` paths.
- **`uv venv --system-site-packages`** — the machine already has torch,
  transformers, peft, trl, datasets, bitsandbytes, accelerate, unsloth, fastapi,
  sqlalchemy and more. Reuse them; installing the `ml` extra into an isolated
  venv would download several GB and the C: drive has limited free space.
- **torch is a CPU-only build.** `torch.cuda.is_available()` is `False` even
  though an RTX 4060 (8 GB) is present. GPU fine-tuning needs a CUDA torch build;
  `/api/capabilities` and `slmforge doctor` surface this rather than failing late.
- **Ollama server is usually not running** (client 0.32.1 is installed). Code must
  degrade gracefully — `OllamaClient.list_models()` returns `[]`, never raises.
- **Windows consoles use a legacy code page**, so writing Unicode (`→`, `—`) to
  stdout raises `UnicodeEncodeError`. `log.force_utf8_stdio()` fixes the streams;
  keep CLI *output* ASCII anyway.
- **Vite 8 may bind IPv6 only** (`[::1]:5173`), so `127.0.0.1:5173` is refused while
  `localhost:5173` works. Serve the built UI from FastAPI (`npm run build`, then
  `slmforge serve`) when you want a single process.
- **Model cache is large.** Keep `HF_HOME` inside the workspace (`./.cache/...`),
  which is gitignored, and set `allow_patterns` when downloading datasets.
- Prerequisites are declared in `.env.example`. Document new settings there too.

---

## 7. Testing expectations

- Add a test with every stage you implement; cover the *contract*, not internals.
- Use in-memory SQLite (`SLMFORGE_DATABASE_URL=sqlite:///:memory:`) for DB tests
  and override the session dependency for API tests.
- Mock external services (`respx` for Ollama/HF/LLM HTTP); never make tests depend
  on a running Ollama, the HF Hub, or a GPU.
- Training/evaluation tests must be runnable on tiny synthetic data with an
  obviously separable task so CPU CI stays fast.
- A test that passes because it asserts nothing is a bug.

---

## 8. Current status & how to pick up work

The foundation (Stage 0) is complete: config, DB + schema, stage contracts,
orchestrator, job queue and worker, sandbox interface, integrations, API,
CLI, web scaffold and docs.

The eight pipeline stages are declared and documented but their `run()` bodies
raise `NotImplementedError` with the milestone that will implement them. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the agenda, exit criteria and order.

When implementing a stage:

1. Read its docstring in `src/slmforge/pipeline/<stage>.py` — it is the spec.
2. Implement the algorithm; keep the persistence out of it.
3. Wire any long-running part as a job handler in `jobs/handlers.py`.
4. Flip `implemented: True` in `api/routes/stages.py::pipeline_overview`.
5. Update the stage docstring, `docs/ROADMAP.md` status and `README.md` if the
   user-facing behaviour changed.
6. Add tests and run `pytest`, `ruff` and `mypy`.
