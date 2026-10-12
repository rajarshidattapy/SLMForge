# SLMForge — what it is and how to run it

*For someone seeing this repo for the first time. Snapshot: 12 October 2026.
Detail lives in [README.md](README.md), [docs/ROADMAP.md](docs/ROADMAP.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## 1. What this is (30 seconds)

**SLMForge turns one English sentence into a task-specific Small Language Model.**

You write something like:

> "Build a small model that classifies customer support tickets into Billing,
> Technical, Account Access or General and returns only a valid JSON object."

SLMForge then walks a pipeline and shows its work at every step:

```
Describe -> Specify -> Prepare Data -> Select Model -> Baseline
        -> Plan -> Fine-tune -> Evaluate -> Deploy -> Callable endpoint
```

Two rules define the product:

1. **Evidence over enthusiasm.** Success is "the adapted model beat the baseline
   on held-out data" — or an honest "don't fine-tune". A training job that exits
   0 proves nothing by itself.
2. **Transparency and consent.** Costly or irreversible steps (downloading data,
   training, deploying) have an approval gate, a pre-flight estimate, and a
   plain-English record of what was chosen and why.

Scope is deliberately narrow: **one honest demo path** (support-ticket
classification), not a production ML platform. "Works only up to the demo state"
is a design decision — every stage must prove itself on a real run before the
next one starts.

---

## 2. What works today (and what does not)

| Area | State |
|---|---|
| Foundation (config, DB, pipeline contracts, job queue + worker, sandbox interface, API, CLI, dashboard shell) | Done |
| Stage 1 — Analyze: prompt to editable `MLSpec` (planner LLM, plus an offline heuristic fallback labelled in `generated_by`) | Done |
| Stage 2 — Curate data: reviewable dataset plan, approve, then a worker build producing `train/validation/test.jsonl` + `DatasetCard` | Done |
| Stage 3 — Select model: ranked shortlist with per-candidate rationale (VRAM fit, licence, locality), approval/rejection gate, alternatives persisted | Done |
| Sandbox: real Docker boundary (network denial, read-only root, non-root user, memory/time limits); subprocess kept as dev fast path | Done |
| Stages 4-8 (baseline, training plan, fine-tune, evaluation, deployment) | Not built — the API answers `501` honestly and the UI marks them "not built yet" |

Last verified live run (real API + worker + Hub + planner LLM): the ticket prompt
produced a 4-label spec, a 5-candidate dataset plan (top fit 0.825), and an
approved build of **3,998 examples** (train 2,799 / validation 400 / test 799)
with hashes and row counts reconciled against the files on disk.

The pipeline also reports uncomfortable facts instead of hiding them: the demo
dataset cannot support an "Account Access" class (`marginal`, error-severity
issue), is 100% synthetic, and is licensed `cc-by-nc-4.0` (fine for the demo,
disqualifying for commercial deployment).

Quality gates: 243 tests, `ruff check`, `ruff format --check` and `mypy` clean
(54 source files); 11 Docker isolation tests pass against a live daemon.

---

## 3. How to run it (5 minutes to a working dashboard)

### Prerequisites

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Node 20+
- `hf` CLI authenticated (`hf auth login`) and `duckdb` where `hf` can see it
  (both on the **global** interpreter, not `.venv`)
- Docker Desktop running (or set `SLMFORGE_SANDBOX_BACKEND=subprocess` for a
  weaker dev-only mode)

### Setup

```bash
git clone <repo-url> SLMForge && cd SLMForge

# Windows (Git Bash) — POSIX uses .venv/bin/python
uv venv --system-site-packages .venv
uv pip install -e '.[dev]' -p .venv/Scripts/python.exe

cp .env.example .env   # then edit as needed; no key required for the offline path
.venv/Scripts/python -m slmforge doctor   # fix what it complains about first
.venv/Scripts/python -m slmforge init-db
docker build -f docker/sandbox.Dockerfile -t slmforge/sandbox:latest .
```

`--system-site-packages` is intentional: the reference machine already carries
torch/transformers/peft, and reinstalling them costs gigabytes.

### Start it (3 terminals)

```bash
.venv/Scripts/python -m slmforge serve     # terminal 1 — API + built UI at http://127.0.0.1:8000
.venv/Scripts/python -m slmforge worker    # terminal 2 — required for any dataset build
.venv/Scripts/python scripts/demo_smoke.py # terminal 3 — optional end-to-end proof (needs network)
```

For UI development instead of the served build:

```bash
cd web && npm install && npm run dev   # http://localhost:5173 (use localhost, not 127.0.0.1)
```

### Click path (the demo, in the UI)

1. Paste the ticket-classification sentence (or "Use the ticket example") and
   **Create build**.
2. In **Pipeline**, **Run** `analyze` — the **Specification** panel shows labels,
   readiness and which path ran (`generated_by`: planner model vs heuristic).
3. In **Dataset**, **Plan dataset** — candidates appear with fit score, rationale,
   row counts, label coverage, licence note and warnings. Nothing downloads here.
   If `huggingface.co` is unreachable, tick **Offline (catalog only)** to skip
   the Hub search — the demo dataset is hand-verified in the catalog, so planning
   still works.
4. Pick a candidate, set the target count, **Approve & build** — this queues a
   `dataset_build` job. Watch it in **Jobs** (polls every 5s; cancel if needed).
5. When the job succeeds, the **Dataset card** shows splits, label distribution,
   synthetic share, sources and issues; **Artifacts** lists the JSONL files under
   `artifacts/<project-id>/dataset/`; **Build log** shows the event trail.
6. In **Pipeline**, **Run** `Select base model` (approval is required even to
   rank), review the shortlist — every candidate states its QLoRA fit, licence
   verdict and reason — then approve one. The project advances to
   `model_selected`; rejecting records your notes and stays put.

### Same path as API calls (scriptable)

```bash
API=http://127.0.0.1:8000
PROJECT=$(curl -s -X POST $API/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Build a small model that classifies customer support tickets into Billing, Technical, Account Access or General and returns only a valid JSON object."}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -s -X POST $API/api/projects/$PROJECT/stages/analyze/run -H 'Content-Type: application/json' -d '{}'
curl -s -X POST $API/api/projects/$PROJECT/stages/curate_data/run -H 'Content-Type: application/json' -d '{}'
curl -s -X POST $API/api/projects/$PROJECT/jobs -H 'Content-Type: application/json' \
  -d '{"kind":"dataset_build","params":{"build":true,"selected":["hf:Tobi-Bueck/customer-support-tickets"],"target_examples":4000}}'
curl -s $API/api/projects/$PROJECT/jobs | python -m json.tool
```

A synchronous build (`stages/curate_data/run` with `{"build": true}`) is refused
with 409 by design — builds download data, so they only run on a worker.
Unbuilt stages answer 501 with the milestone that owns them.

### Optional: Sign in with Hugging Face (local)

Sign-in is off by default and nothing in the demo needs it — Hub data access
goes through `hf auth login` on the CLI, not through this. It only identifies
who is using the build. To enable it locally (no public URL needed — the
redirect happens inside your own browser):

1. Create an app at `huggingface.co/settings/applications/new`:
   - Homepage URL: `http://localhost:8000` (display-only link on the consent screen)
   - Redirect URI: `http://localhost:8000/api/auth/huggingface/callback` (must match exactly)
2. In `.env`:
   ```
   HF_CLIENT_ID=<from the HF app page>
   HF_CLIENT_SECRET=<from the HF app page>
   SLMFORGE_PUBLIC_BASE_URL=http://localhost:8000
   ```
3. Restart `serve`, then use the **Sign in with Hugging Face** button in the
   sidebar (or visit `http://localhost:8000/api/auth/huggingface/login`
   directly). Verify at `http://localhost:8000/api/auth/me`.

Rules that bite: use `localhost`, not `127.0.0.1`, in the browser — they are
different cookie hosts, so signing in on one is invisible on the other. The
redirect URI must equal `SLMFORGE_PUBLIC_BASE_URL` +
`/api/auth/huggingface/callback` character-for-character or HF rejects the
callback. On a deployed backend, `SLMFORGE_PUBLIC_BASE_URL` must instead be the
public URL.

---

## 4. Where things live

- `src/slmforge/pipeline/` — the eight stages (`analyzer.py`, `datasets.py` done;
  the rest are honest stubs). Stages never touch the DB; the orchestrator does.
- `src/slmforge/jobs/` — durable queue + worker + handlers (long work lives here,
  never in the API process).
- `src/slmforge/sandbox/` — `subprocess_backend.py` (dev fast path) and
  `docker_backend.py` (real boundary).
- `src/slmforge/integrations/` — `hf_cli.py` is the primary Hub interface;
  every network call retries via `net.retry_call`.
- `src/slmforge/api/` — thin routers (projects, stages, jobs, deployments, health).
- `web/src/` — dashboard: prompt form, capability bar, pipeline, spec, dataset
  review, jobs, build log, artifacts. Served from `web/dist` by FastAPI.
- `scripts/demo_smoke.py` — the live end-to-end check the unit suite cannot do.

---

## 5. Constraints you will hit (so they do not surprise you)

- torch here is CPU-only (`cuda.is_available() == False`) despite the RTX 4060 —
  fine for the data demo, too slow for real training until a CUDA wheel lands.
- The HF account has no credits: GPU training and hosted endpoints are
  implemented-but-gated paths, not defaults. The demo trains/serves locally.
- The free planner model is intermittently overloaded (retried with backoff, then
  falls back to the heuristic and says so).
- The reference network resets TLS to `huggingface.co` under load — everything
  retries; profile remotely once, then materialise locally once.
- Vite 8 may bind IPv6 only: use `localhost:5173`, not `127.0.0.1:5173`.
- `localhost` and `127.0.0.1` are different cookie hosts: pick `localhost`
  everywhere (including HF sign-in) or sessions disappear across hosts.

Next up per [docs/ROADMAP.md](docs/ROADMAP.md): Stage 3 (model discovery and
ranking), then the baseline harness. Status table: [README.md](README.md).
