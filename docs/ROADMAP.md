# SLMForge Development Agenda

An incremental plan from the current foundation to a demonstrable end-to-end
prototype. Each stage is independently shippable and has explicit **exit
criteria** — a stage is done when its criteria pass on a real run, not when the
code is written.

Guiding rule for sequencing: **the happy path must be demonstrable at the end of
every stage.** We never build a beautiful UI for a pipeline that cannot run.

---

## Status at a glance

| Stage | Theme | Status |
|---|---|---|
| 0 | Foundation | ✅ Complete |
| 1 | Prompt → `MLSpec` (analyzer) | ✅ Complete |
| 2 | Dataset curation | ✅ Complete |
| 3 | Model discovery & ranking | ✅ Complete |
| 4 | Docker sandbox + baseline | ✅ Complete |
| 5 | Training planner | ⬜ Not started |
| 6 | Fine-tuning execution | ⬜ Not started |
| 7 | Evaluation & comparison | ⬜ Not started |
| 8 | Deployment to a hosted endpoint | ⬜ Not started |
| 9 | Dashboard, retention, hardening | ⬜ Not started |

Dependency order is essentially linear, with one shortcut: **Stage 4's sandbox
work can proceed in parallel with Stage 2/3**, and Stage 9's dashboard work can
be pulled forward at any time since the API is already complete enough to drive.

---

## Stage 0 — Foundation ✅ Complete

**Goal:** a runnable, testable skeleton that makes later stages small changes.

Delivered:

- Validated configuration (`config.py`) with `.env.example` covering every knob.
- Database + six-table schema (`models.py`) with WAL-mode SQLite.
- The eight stage contracts in `schemas/` (`MLSpec`, `DatasetPlan`, `ModelCandidate`,
  `TrainingPlan`, `EvalReport`, `DeployRecord`, …).
- `Stage` abstraction, `Orchestrator` with events and status transitions, and an
  ordered stage registry.
- Durable job queue with leases + worker loop + job handlers.
- `Sandbox` interface and a working subprocess backend.
- Ollama, Hugging Face and planner-LLM integrations (graceful when offline).
- FastAPI control plane: projects, pipeline, spec review, stage runs, deployments,
  capabilities/health.
- Vite + React + TypeScript dashboard (capabilities, prompt form, pipeline, spec,
  build log, artifacts).
- CLI (`serve`, `worker`, `init-db`, `doctor`), 61 tests, ruff + mypy clean.

Exit criteria — all met:

- `pytest`, `ruff check`, `mypy` pass.
- `slmforge serve` boots, serves the API and the built SPA, and `/api/capabilities`
  correctly reports this machine's limits.
- A project can be created, inspected, given a spec, and the honest `501` for an
  unbuilt stage reaches the UI.

---

## Stage 1 — Prompt → `MLSpec` (the analyzer) ✅ Complete

**Goal:** turn a sentence into a reviewed, editable specification.

Delivered:

- `AnalyzerStage` with both paths — the planner LLM (JSON schema, validation, repair)
  and a deterministic heuristic — never blended, always labelled in `generated_by`.
- Prompt template plus support, ambiguity and high-stakes gates; out-of-scope
  requests are refused with a reason.
- Label extraction from quoted terms **and** unquoted enumerations ("…into Billing,
  Technical, Account Access or General…"), which is how the demo prompt is written.
- Output-shape questions ("what should the JSON look like?") are answered with the
  canonical single-label shape and recorded as an assumption, so a runnable build is
  not blocked on a convention the platform already decided. Questions that really do
  change the contract still block.
- Planner-LLM failures are retried with backoff, including the OpenRouter case of an
  error payload delivered with HTTP 200; a genuinely bad reply still falls back.

Exit criteria — met, verified live: the ticket prompt yields four labels,
`task_type=classification`, `output.format=json_object`, a primary metric and
`readiness=ready_for_review`; an ambiguous prompt still asks; an out-of-scope prompt is
refused honestly.

---

## Stage 2 — Dataset curation ✅ Complete

**Goal:** from spec to real, trustworthy train/validation/test files.

Delivered:

- Two modes in one stage, selected by `params`: **plan** (discovery, scoring, one
  remote DuckDB profile, no download) and **build** (`params={"build": true}`).
  A plan persists on the project, reports `needs_approval`, and deliberately does
  **not** advance it to `data_ready`; the orchestrator now refuses to advance past a
  stage that is still waiting for a decision.
- Discovery: a hand-verified curated catalog plus Hub search leads, each with a
  transparent four-term fit score and a rationale string. Non-catalogued hits are
  offered as leads only — no column guessing.
- Materialisation: one projected remote read per source (`SELECT concat_ws(...)`),
  written locally once, then worked on entirely offline. Retried, and retry-safe
  (a half-written file cannot be read back as success).
- Label mapping onto the spec's labels, as an audit trail: every raw value is `keep`,
  `map` or `drop`, each with a reason, and a mapping onto a label the spec does not
  list is dropped rather than forced.
- Quality: empty labels, too-short text, exact duplicates, suspected PII (counted as a
  *kind*, never copied into the report), class balance, and a near-duplicate leakage
  check across splits that removes leaks and says how many.
- Deterministic, label-stratified, seeded splits; byte-identical output on a re-run
  (asserted on the files themselves).
- Uploads (JSONL/CSV with a column mapper) treated as untrusted: type whitelist, size
  cap, row cap, `errors="replace"` decoding.
- A `DatasetCard` with per-source provenance, licences, synthetic fraction, file
  hashes and the quality issues; every number in it describes the files that were
  written, so totals equal the split sizes.
- `target_examples` is an approval boundary: the build delivers that many at most,
  allocated proportionally so the source's class balance survives.
- Long builds are queued: `POST /api/projects/{id}/jobs` (kind `dataset_build`),
  claimed and run by the worker; the synchronous endpoint refuses a build with a 409.

Exit criteria — met, verified live: the ticket prompt produced a plan (5 candidates,
fit 0.825), an approved build of 3,998 examples (train 2,799 / val 400 / test 799) from
**`Tobi-Bueck/customer-support-tickets`**, and a card that reports the two leaks it
removed, the cc-by-nc-4.0 licence, the 100% synthetic share, and the one class the
dataset cannot support.

### What the live run taught us (recorded, not hidden)

1. **The demo dataset cannot support "Account Access".** Its `queue` column holds 52
   values (Technical Support, Product Support, Customer Service, Billing and Payments,
   plus a long tail of generic content categories); none maps onto account or login
   issues. The plan and the card both say so, and the verdict is `marginal` rather than
   a flattering `sufficient`. Options: re-specify with three labels, or upload examples.
2. **The licence is `cc-by-nc-4.0`** — non-commercial. It is fine for the demo and
   disqualifying for a commercial deployment; the plan surfaces that as a warning.
3. **The dataset is synthetic end to end** (`synthetic_fraction = 1.0`), so the
   evaluation must not treat it as ground truth. A small real held-out set stays on the
   "still to decide" list below.
4. **The Hub CLI is inconsistent** (`datasets parquet` prints a table unless asked for
   JSON; `datasets info` rejects the flag). The client now tries, adapts, and only then
   reports a parse failure — pinned by tests, because getting it wrong breaks every
   build.
5. **One session per job.** Completing a job through a second session writes nothing,
   leaving a finished job reading as `running`. There is now a regression test for it.

---

## Superseded planning notes (kept for context)

### Stage 1 — original plan

**Goal:** turn a sentence into a reviewed, editable specification.

Work:

1. `AnalyzerStage.run()` — call the planner LLM with a JSON schema derived from
   `MLSpec`; validate with pydantic; repair malformed JSON (fence/prose stripping
   already exists in `integrations/llm.py::parse_json_object`).
2. Prompt template covering task type inference, I/O contract, label
   canonicalisation, metric selection, and explicit `assumptions` /
   `open_questions` / `out_of_scope`.
3. Clarifying-question logic: if required slots are missing, set
   `readiness=needs_clarification` and populate `open_questions` instead of guessing.
4. Support gate: map onto `TaskType`; anything outside the MVP set returns
   `unsupported` with a clear explanation rather than a bogus spec.
5. High-stakes detection (medical/legal/financial/employment) → `is_high_stakes`.
6. Deterministic fallback when no planner LLM is reachable: a rule-based spec for
   classification/JSON tasks, clearly labelled `generated_by="heuristic"`.
7. UI: render the spec as an editable form (labels, metrics, constraints) with a
   "Re-analyze" action; save via `PUT /spec`.

**Exit criteria**

- The ticket-classification prompt returns a valid `MLSpec` with four allowed
  labels, `task_type=classification`, `output.format=json_object`, and a primary
  metric — using a locally running Ollama model with no network access.
- An ambiguous prompt produces `open_questions` rather than an invented answer.
- An out-of-scope prompt (e.g. "write my novel") is refused honestly.
- Spec edits persist and survive a re-analyze that the user cancels.

**Risks:** LLM output drift → mitigation: strict schema + validation + repair +
the heuristic fallback. `MLSpec` churn → mitigation: the eval-report contract is
the only downstream dependency, so keep changes additive.

---

### Stage 2 — original plan

**Goal:** from spec to real, trustworthy train/validation/test files.

Work:

1. **Discovery** — derive queries from the spec; search HF datasets + a curated
   catalog; score fit (task, labels, language, domain) *with a rationale*; record
   licence and commercial-use risk; surface candidates as a reviewable
   `DatasetPlan` (never auto-committing).
2. **Upload** — accept CSV/JSONL with a column mapper; treat as untrusted
   (size/type/row limits, encoding checks).
3. **Build** — download with `allow_patterns`; normalise to a canonical record;
   map raw labels onto `allowed_labels` with an auditable `LabelMapping`; drop
   unmappable rows with reasons.
4. **Quality** — missing labels, schema mismatch, near-empty text, suspected PII,
   exact + near-duplicate detection (MinHash or embeddings), class-balance report.
5. **Splits** — deterministic seed-based splits with a leakage check across splits.
6. **Synthesis (optional)** — only when data is insufficient, only with explicit
   approval, always labelled `synthetic=True` and never counted as ground truth.
7. **Materialise** — write `train/validation/test.jsonl` + `DatasetCard` with
   per-source provenance and file hashes; record everything as `Artifact` rows.
8. **Sufficiency verdict** — `sufficient` / `marginal` / `insufficient`, and when
   insufficient, ask the user for examples instead of proceeding on rubbish.

**Still open from this plan (Stage 2.5, not blocking the demo):**

- Embedding/MinHash near-duplicate detection beyond word shingles (shingles are
  cheap and caught 2–7 real leaks; a semantic check would catch more).
- Optional, explicitly approved synthetic top-up for a class with no source data; the
  current answer is "no source rows for that label", which is the honest one.
- Cross-source label-map editing in the UI (the API already accepts an edited plan).
- Schema-mismatch reporting for uploads (the mapper names the columns; it does not yet
  validate types).

**Risks:** irrelevant datasets → mitigation: fit rationale + explicit approval.
Licence encumbrance → mitigation: `is_commercial_friendly` + warning surfaced at
plan review. Slow downloads on limited disk → mitigation: one projected read per
source plus a configurable cache location.

---

## Stage 3 — Model discovery & ranking ✅ Complete

**Goal:** a defensible shortlist, and a chosen base model the user approves.

Delivered:

- `ModelSelectionStage` with two calls, mirroring the dataset plan/approve
  split: ranking (read-only, returns `needs_approval`, advances nothing) and a
  decision (`params={"selected": id}` approves and advances to
  `model_selected`; `{"rejected": true}` records the refusal and stays put).
- Three sources: a hand-verified 7-model catalog (works offline), local marks
  from the Ollama inventory and the HF cache snapshot dirs, and Hub search
  leads offered as unverified, unapproved leads — a dead network degrades to
  catalog-only rather than failing.
- Transparent rubric (task fit .35 / VRAM fit .35 / licence .20 / locality .10)
  with per-candidate `rank_rationale`; ties break toward the cheaper fit, and
  the baseline stage will resolve capability ties with measurements.
- Documented QLoRA heuristic (`1.5 + 1.55 GB` per billion params); oversized
  picks stay blocked (physics), flagged licences are allowed (the approval
  *is* the review).
- 15 contract tests, all services mocked; the approval-gate API tests now run
  against the real stage instead of a monkeypatched stub.

Exit criteria — met, verified live: the ticket spec at 8 GB VRAM ranks 7
candidates that all fit QLoRA, each with a licence verdict and rationale;
`SmolLM2-360M-Instruct` tops on cost among equals; approving advances the
project to `model_selected` with alternatives persisted; rejecting is recorded.

Work (as built):

1. Enumerate Ollama models (`/api/tags`) and the local HF cache.
2. Search the HF Hub for small instruction-tuned candidates, filtered by a
   parameter budget derived from `SLMFORGE_MAX_VRAM_GB`.
3. Normalise metadata: parameter count, context window, licence, quantisation,
   `FineTuneSupport` (methods + min VRAM + tooling).
4. Rank with a transparent rubric (task fit, VRAM fit, licence clearance,
   deployability) producing `rank_score` **and** `rank_rationale`.
5. Flag risky candidates (restricted/unclear licence, too large, no fine-tune
   support) as warnings rather than silently dropping them.
6. `ModelSelection` requires explicit approval; persist alternatives considered.

**Exit criteria**

- Given the ticket spec and 8 GB VRAM, the shortlist contains only models that
  actually fit for QLoRA, each with a licence verdict and a stated reason.
- The chosen model is either present locally or has a verified download path.
- Rejecting the recommendation is possible and recorded.

**Risks:** Hub metadata gaps → mitigation: never trust a missing licence (mark it
`unknown`, require review). Ranking feel → mitigation: rationale strings, and a
test asserting a known-good model outranks an oversized one.

---

## Stage 4 — Sandbox hardening + baseline ✅ Complete

**Goal:** isolate untrusted execution properly, then measure candidates *before*
training.

Delivered (harness; Docker backend already landed earlier):

- `BaselineStage` run exclusively through the `baseline_eval` job (the sync
  endpoint answers 409, so inference never runs in the API process).
- Worker-process GPU inference (CUDA torch, greedy, batch-1) with per-sample
  latency, peak VRAM, seeded subset of the held-out test split, and the same
  system prompt as the training data. The slim Docker image has no ML stack by
  design, so GPU work lives in the worker; dataset input is parsed as data.
- Strict output parsing (bare labels are invalid, fences stripped), accuracy /
  macro-F1 / per-class / confusion, valid-output rate, schema violations, and a
  capped error-analysis sample set — one `EvalRun` per model plus a
  reproducibility bundle (repo, snapshot revision, device, prompt, decoding,
  seed, indices).
- Cheapest-first honesty: unloadable models fail the stage (selected) or are
  recorded with errors (extras) — never a fabricated score.
- 13 contract tests, all inference stubbed; determinism pinned by a real
  `generate()` test after the live run proved one seed per run is not enough
  (`generate()` consumes RNG state even when greedy — the harness reseeds per
  sample).

Exit criteria — met, verified live on the RTX 4060: SmolLM2-360M measured on
CUDA (revision pinned, 0.7 GB peak), two seeded runs agree token-for-token, a
bogus model id fails loudly, `baseline_eval` enqueues (201) while the sync
endpoint still 409s. The Docker `deny` isolation proof stays with the existing
sandbox tests.

Work (as built):

1. **Docker sandbox backend** — image with pinned deps, no host network by default,
   read-only base layer, writable work dir, CPU/memory/time/GPU limits, non-root
   user. Keep `SubprocessSandbox` as the dev fast path and be explicit in the UI
   about which is active.
2. Network policy enforcement (`deny` / `allowlist` / `allow`) for downloads vs.
   offline evaluation.
3. Baseline harness: fixed evaluation subset, controlled prompt, greedy decoding,
   recorded seed; parse/validate outputs against the spec's schema; compute task
   metrics, valid-output rate, latency percentiles and peak VRAM.
4. Run it as a `baseline_eval` job with streamed logs and an `EvalRun` artifact.
5. Cheapest-first ordering: any candidate that cannot even produce valid output is
   reported, not hidden.

**Exit criteria**

- Baseline numbers reproduce: two runs with the same seed agree within tolerance.
- A deliberately broken "model" produces a reported failure, not a fabricated score.
- The Docker backend blocks a script that attempts an outbound network call when
  the policy is `deny`.
- Baseline runs never execute in the API process (verifiable by process tree).

**Risks:** Docker unavailable on the demo machine → mitigation: the subprocess path
still works, with the isolation caveat stated in the UI. GPU passthrough complexity
→ mitigation: CPU-only baseline is acceptable for the demo and much simpler.

---

## Stage 5 — Training planner

**Goal:** decide *whether* to fine-tune and *how*, with a cost estimate before
any money is spent.

Work:

1. **Gating** — if the baseline already meets the acceptance criteria, or the task
   is better served by prompting/RAG, recommend `no_finetune` / `prompt_only` /
   `rag` with reasoning. This is a success path, not a failure.
2. **Method selection** — rules for `qlora` (tight VRAM), `lora` (base fits in
   memory), `sft` (small model, full fine-tune affordable).
3. **Hyperparameters** — LR from method/size, epochs from dataset size, `max_seq_len`
   from the token-length distribution, LoRA targets from the model architecture's
   module names. Documented heuristics, not magic numbers.
4. **Estimation** — VRAM, disk, wall-clock, GPU-hours and cost from parameter count
   and token budget; compare against `SLMFORGE_MAX_VRAM_GB` / `MAX_TRAIN_MINUTES`
   and warn on overrun.
5. **Script generation** — render a standalone, reproducible training script into
   the project artifacts; store `config_fingerprint` (hash of script + config +
   data hash).
6. **Approval gate** — present estimate + warnings; block without approval.

**Exit criteria**

- For a 0.5B model on 8 GB VRAM with ~2k examples, the plan proposes QLoRA with
  sane hyperparameters and an estimate within 30% of the observed run.
- A task where prompting already passes acceptance produces a *"don't fine-tune"*
  recommendation with evidence attached.
- The generated script runs standalone given only the data files and its config.

**Risks:** wrong estimates → mitigation: mark `confidence`, compare estimate vs.
actual after each run, and show both. Over-eager fine-tuning → mitigation: the
Stage 7 comparison can and must say "keep the baseline".

---

## Stage 6 — Fine-tuning execution

**Goal:** run the approved script safely, with visible progress and no false success.

Work:

1. Materialise script + config into the job work dir; run in the sandbox with the
   resolved limits (GPU allowed, otherwise fail fast with a clear message rather
   than silently training on CPU for 12 hours).
2. Stream stdout/stderr to the job log; parse structured progress
   (step/loss/LR/VRAM) for the dashboard.
3. Persist checkpoints and the final adapter as `Artifact` rows; record peak VRAM,
   wall clock, final loss, loss curve.
4. Cancellation: SIGTERM → grace period → SIGKILL, marking the job `cancelled`.
5. Retry/resume semantics: a failed run can be retried; attempts and the reason for
   each are recorded.
6. **Never report success on a non-zero exit code**, even if a checkpoint exists.

**Exit criteria**

- A QLoRA run on the ticket classification task completes end to end and produces
  an adapter + `TrainingRecord` with a loss curve.
- Cancelling mid-run stops the process, leaves no orphaned GPU memory, and marks
  the job `cancelled`.
- A deliberately crashed run reports `failed` with the captured stderr tail.
- Peak VRAM is measured, not guessed, and is compared against the Stage 5 estimate.

**Risks:** OOM → mitigation: QLoRA defaults, gradient checkpointing, and shrinking
batch/grad-accum automatically once before failing. Environment mismatch (CPU-only
torch) → mitigation: detect at plan time and block with instructions.

---

## Stage 7 — Evaluation & comparison

**Goal:** prove — or honestly disprove — that fine-tuning helped.

Work:

1. Re-run the **same** harness as Stage 4 on the held-out test split.
2. Task metrics + valid-output rate + schema violations.
3. Latency percentiles, peak VRAM, cost per 1k calls.
4. Per-class confusion table and an error-analysis sample set for the UI.
5. Comparison with significance testing (bootstrap CI / McNemar) so small-sample
   noise is not sold as an improvement.
6. Verdict: `accept_finetuned` / `keep_baseline` / `inconclusive` / `needs_more_data`,
   with limitations and recommendations, and a `best_alternative` when fine-tuning
   did not pay off.

**Exit criteria**

- Baseline vs. fine-tuned tables appear in the UI with deltas and confidence.
- A run on deliberately noisy data returns `inconclusive` or `keep_baseline` rather
  than a flattering number.
- Every reported number names its split, sample count and metric definition.

**Risks:** metric gaming → mitigation: multiple metrics + format-validity checks +
held-out data + significance. Small test sets → mitigation: report sample counts
and widen intervals instead of asserting a win.

---

## Stage 8 — Deployment to a hosted endpoint

**Goal:** the product's promise — a callable endpoint, on Hugging Face, with auth.

Work:

1. Merge the adapter (optional) into the base model; tag the version.
2. Push model/adapter to the HF Hub (private by default, `SLMFORGE_HF_ORG`),
   including a model card: eval summary, metrics, known limitations, licence.
3. Create the endpoint per `SLMFORGE_DEPLOY_TARGET`:
   - `hf_endpoint` — HF Inference Endpoint on the selected instance type;
   - `hf_space` — Gradio/FastAPI Space for demo traffic;
   - `local_openai` — OpenAI-compatible server on the local GPU.
4. Provision an API key; store **only** a reference (`Deployment.auth_ref`).
5. Return a request + response example, model version, eval summary, limitations.
6. Usage/latency/error reporting, plus **stop** (and delete) to end billing.
7. Hard approval gate and a pre-flight cost/quota check — never auto-deploy.

**Exit criteria**

- A real endpoint answers a real request with the documented schema, once, from the
  UI, with the key stored as a reference only.
- Stopping the endpoint releases the resource and updates status to `stopped`.
- A licence that forbids commercial hosting blocks deployment with an explanation.
- Graceful failure with a readable error when HF credentials or quota are missing.

**Risks:** cost/quota surprises → mitigation: pre-flight estimate + approval +
stop control. Cold starts → mitigation: document them in the deployment record
rather than hiding latency.

---

## Stage 9 — Dashboard, retention & hardening

**Goal:** make the build record trustworthy and the deployment safe to leave running.

Pulled forward already (the API was complete enough to drive): the dataset
review panel (plan → scored candidates → approve-and-build, with an offline
catalog-only toggle), the worker-jobs panel (polling, cancel), spec/build-log/
artifact tabs, and the HF sign-in chip. Still ahead: everything under "Work".

Work:

1. Full build-record UI: side-by-side baseline vs. fine-tuned comparison charts,
   error-analysis table, artifact browser with hashes, per-stage timings.
2. Retention & deletion: delete uploads/artifacts and generated models, with a
   documented retention policy and confirmation.
3. Cost and usage views; endpoint health checks and automatic stop of idle endpoints.
4. Rate limiting, request quotas and API keys for the SLMForge API itself.
5. Alembic migrations; a Postgres path for multi-user deployments.
6. Observability: structured logs, job/stage metrics, and a failure digest.
7. Accessibility and responsive polish on the dashboard.

**Exit criteria**

- Every claim on the dashboard links to the artifact or recorded event that
  supports it.
- Deleting a project removes its artifacts from disk, verifiably.
- Migrations take an existing database to head with no data loss.

---

## Cross-cutting workstreams (continuous)

| Workstream | Practice |
|---|---|
| **Tests** | Every stage ships with contract tests on tiny synthetic data; external services mocked. |
| **Reproducibility** | Seeds, config fingerprints, script hashes and dataset hashes recorded for every run. |
| **Security** | Untrusted input stays in the sandbox; secrets never enter artifacts, logs, the DB or the repo. |
| **Honesty** | No metric shown without its split and sample count; negative and inconclusive results are first-class. |
| **Docs** | Stage docstrings and this file are updated in the same change as the code. |
| **Licensing** | Every dataset and model records licence + commercial-use verdict before use. |

---

## Decisions (settled)

Every decision below is now encoded in `config.py` / `.env.example`.

1. **Planner LLM — OpenRouter, free models.** One `OPENROUTER_API_KEY`, OpenAI-compatible
   `/chat/completions`, so `integrations/llm.py` needed no rewrite. Verified against the
   live catalogue: 469 models, **19 free**, including `nvidia/nemotron-3-ultra-550b-a55b:free`
   (1M ctx, used for the spec and the training-method choice) and
   `google/gemma-4-26b-a4b-it:free` (used for bulk synthetic generation).
   The deterministic heuristic fallback stays, so the pipeline demos with no key.
2. **Demo task — ticket classification, single happy path.** Dataset verified live:
   **`Tobi-Bueck/customer-support-tickets`**, 61,765 rows, public and ungated,
   one parquet file, columns `subject, body, answer, type, queue, priority, language,
   version, tag_1..tag_8`. **It is synthetic** ("Synthetic IT Ticket Generator",
   "no PII"), so it must be labelled as such and must not be the only thing we
   evaluate on. Two facts learned when building on it, both surfaced automatically:
   its licence is **`cc-by-nc-4.0` (non-commercial)**, and its `queue` column cannot
   produce an **Account Access** class (52 raw values, none about accounts or logins),
   so the four-label spec is only three-quarters supported by this source.
3. **Sandbox — two tiers, activated separately.** `docker` runs untrusted work locally
   with real filesystem/network isolation; `hf_jobs` runs it in managed ephemeral HF
   containers, including GPUs. Both sit behind the existing `Sandbox` interface.
4. **Deployment — local by default.** This HF account has no credits, so the default
   `deploy_target` is `local_openai`: an OpenAI-compatible server hosting the
   fine-tuned model. `hf_endpoint` / `hf_space` remain implemented paths, switched on
   when credits exist.
5. **Data — Hugging Face first, via the `hf` CLI.** Discovery, parquet URLs, DuckDB
   profiling, Jobs and endpoints all go through the official `hf` CLI, which also
   carries its own authentication (`hf auth login`) so the app never holds the token.

## Open blockers

1. **No HF credits.** `hf auth whoami` reports `isPro: false`, `canPay: false`,
   `billingMode: prepaid`. `hf_jobs` (GPU training) and `hf_endpoint` (hosted
   endpoint) both need a paid plan or prepaid credits. Until then: train locally,
   serve locally.
2. **Local GPU is active.** `.venv` carries torch `2.11.0+cu128` (plus a matching
   `torchvision`, shadowing the global CPU builds), so `torch.cuda.is_available()`
   is `True` on the RTX 4060 (8 GB). Baseline and training measure real GPU
   numbers; the Docker sandbox image stays CPU-only by design.

### Free alternatives to Hugging Face paid compute

Searched rather than assumed. None of these need a card; all are for the **demo**
only, so they stay behind the same `Sandbox` interface rather than becoming a
hard dependency.

| Option | GPU | Quota | Automatable? | Verdict |
|---|---|---|---|---|
| **Local RTX 4060 (8 GB)** | Yes | Unlimited | Best | **First choice.** One free CUDA wheel away; no queue, no upload, no data leaves the machine. |
| **Kaggle Notebooks** | T4 or P100, 16 GB VRAM | ~30 GPU-hours/week, 12 h sessions, stated and visible | Yes — `kagglehub` / `kaggle` CLI runs a notebook, polls status, pulls artifacts back | **Best free cloud option.** A stated quota is far more predictable than Colab's. |
| **Lightning AI Studio** | T4/L4-class | ~80 free GPU hours | Yes, via its API/CLI | Good fallback if the Kaggle quota runs out mid-week. |
| **Google Colab** | T4/P100 | 15–30 h/week, *availability-based*, undocumented | Poorly — no stable job API, cells time out | Only for manual exploration, not for the pipeline. |
| **HF Jobs `cpu-basic`** | — | $0.01/hr | Yes (`hf jobs`) | Cheap but not free; useful only to prove the Job path end-to-end. |

Recommended path: **train on the local 4060**; if that is unavailable, submit the
same generated training script to **Kaggle** as a managed job. Both reuse the
`hf_cli.jobs_*`-style shape (submit → poll → pull artifacts), so a `kaggle` backend
is a small addition to the `Sandbox` abstraction, not a rewrite.

Also worth noting for the *serving* half: it needs no GPU at all — a small
OpenAI-compatible FastAPI server hosting the fine-tuned model on CPU is enough to
deliver the demo's "callable endpoint".
3. **The reference network resets TLS handshakes** with `huggingface.co` under load —
   observed repeatedly from both the `hf` CLI and DuckDB, sometimes failing several
   attempts in a row. Mitigated by `net.retry_call` (backoff + jitter, clear
   `TransientNetworkError`), and by profiling remotely once and then materialising a
   single local copy rather than re-querying the Hub in a loop.
4. **Docker is running now**, and the `docker` sandbox backend is delivered with its
   isolation tests passing (network denial, read-only root, non-root user, memory
   ceiling, timeout cleanup). The image is `slmforge/sandbox:latest` (1.03 GB), built
   from `docker/sandbox.Dockerfile`.
5. **The free planner model is intermittently overloaded** (OpenRouter forwards a
   provider 503 as a *200 response with an error payload*). Retried with backoff now;
   when it is exhausted the analyzer falls back to the heuristic and says so. The
   demo therefore does not depend on the planner being up.

## Still to decide

- **Serving backend.** `local_openai` via a small OpenAI-compatible FastAPI server
  (no extra tooling, works today) versus exporting GGUF and registering the model with
  the already-installed Ollama (nicer DX, but adds a conversion step and llama.cpp).
- **A small real held-out set.** The verified dataset is synthetic; a hand-collected
  set of real tickets would keep the evaluation honest rather than grading
  synthetic-with-synthetic.

---

## Definition of done for the prototype

The MVP is demonstrable when, from one sentence, a user can:

1. get a specification they can edit,
2. see the datasets and the model that were chosen and why,
3. read a baseline,
4. approve and watch a fine-tune run inside the sandbox,
5. see an honest before/after comparison — including "it did not help" when true,
6. click deploy and call a hosted endpoint.
