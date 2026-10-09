# SLMForge: Build Task-Specific Small Language Models from a Prompt

> **One-line description:** SLMForge turns a natural-language
> requirement into a task-specific Small Language Model (SLM) by
> automating data discovery, model selection, fine-tuning, evaluation,
> and deployment.

## 1. Project Overview

SLMForge is an AI development platform inspired by the simplicity of
prompt-to-app builders such as Lovable. Instead of generating a web
application from a prompt, SLMForge helps users create a specialized
language model.

A user describes the model they need in plain language. SLMForge
interprets the request, converts it into a structured machine-learning
specification, discovers or prepares suitable training data, selects a
compatible base model, plans an appropriate adaptation strategy, runs
experiments in an isolated environment, evaluates the results, and
packages the selected model behind an API endpoint.

The central idea is to hide the operational complexity of the
machine-learning lifecycle without hiding the decisions that matter.
Users should be able to inspect the proposed dataset, model, training
approach, evaluation results, cost estimates, and deployment settings
before committing to expensive or irreversible steps.

**Core workflow:**

**Describe → Specify → Prepare Data → Select Model → Establish Baseline
→ Fine-tune → Evaluate → Deploy**

SLMForge is intended to make task-specific SLM development more
approachable for developers, startups, students, and organizations that
need a focused model but do not have a dedicated ML infrastructure team.

## 2. The Problem

Building a specialized language model is a multi-stage process that
currently requires knowledge of machine learning, dataset preparation,
model architectures, fine-tuning methods, GPU infrastructure,
evaluation, and deployment.

A developer may know exactly what they want an AI system to do, but
still need to: - Translate the requirement into a well-defined ML
task. - Find or create relevant training examples. - Clean, validate,
deduplicate, and format the data. - Compare suitable base models and
their licenses or hardware requirements. - Choose between prompting,
retrieval, fine-tuning, or other approaches. - Configure training
libraries, hardware, and hyperparameters. - Measure whether the adapted
model actually improves on the base model. - Package, host, monitor, and
expose the model through an API.

These steps can be difficult to coordinate, especially for people
without specialized ML experience. Poor choices at any stage can waste
compute, produce unreliable results, or lead to a model that performs
worse than the original.

Many AI development tools make it easier to call general-purpose models,
create prompts, or build applications around them. SLMForge focuses on a
different goal: **automating the process of creating and delivering a
task-specific model**, where fine-tuning is useful and justified.

## 3. The Proposed Solution

SLMForge provides a single workflow that coordinates the steps needed to
create a specialized SLM.

The user starts with a request such as:

> "Build a small model that classifies customer support tickets into
> billing, technical issues, account access, and general inquiries.
> Return only a valid JSON object containing the category and
> confidence."

From this request, the platform derives the task, expected inputs and
outputs, constraints, evaluation criteria, and likely model
requirements. It then searches for relevant datasets and candidate
models, establishes a baseline, and selects a training strategy based on
the task, data quality, model compatibility, and available compute.

After training, SLMForge compares the adapted model with the baseline
using appropriate metrics. It reports the results and, with the user's
approval, deploys the chosen model as a hosted endpoint.

The platform should not blindly fine-tune every request. For some use
cases, a carefully designed prompt, structured output, or
retrieval-augmented generation (RAG) may be more effective, cheaper, or
easier to maintain. SLMForge should identify these cases and explain the
trade-off rather than training a model unnecessarily.

## 4. Target Users

### Primary users

-   **Developers and independent builders:** People who need a
    specialized model but do not want to build an ML pipeline from
    scratch.
-   **Startups and small teams:** Teams that want task-specific
    behavior, predictable interfaces, and control over inference costs.
-   **Students and researchers:** Users who want to experiment with
    datasets, SLMs, and fine-tuning without manually assembling the
    entire workflow.

### Potential organizational users

-   Customer support teams classifying or routing tickets.
-   E-commerce businesses extracting structured information from product
    descriptions or customer messages.
-   Engineering teams categorizing bug reports or summarizing
    code-review feedback.
-   Operations teams converting repetitive text into structured records.
-   Organizations that need a compact model for a clearly defined
    internal workflow.

Use cases involving medical, legal, financial, or other high-impact
decisions would require stricter validation, oversight, privacy
controls, and domain-specific evaluation. SLMForge should not imply that
a model is safe or suitable for such uses solely because it has
completed training.

## 5. Example User Journey

### Step 1: Describe the requirement

The user enters a natural-language request:

> "Create a lightweight model that reads customer support tickets and
> assigns each ticket to Billing, Technical, Account Access, or General.
> It should return valid JSON and prioritize classification accuracy."

The user may optionally specify latency limits, budget, deployment
preferences, privacy constraints, example inputs and outputs, or a
preferred model family.

### Step 2: Convert the prompt into an ML specification

SLMForge identifies the task and proposes a structured specification:

-   **Task:** Text classification with structured output.
-   **Input:** Customer support ticket text.
-   **Output:** One allowed category in a JSON response.
-   **Success criteria:** Classification quality, valid output format,
    latency, and inference cost.
-   **Constraints:** Model size and available compute, if specified.
-   **Evaluation:** Held-out labelled examples and task-appropriate
    metrics.

The user can review and edit this specification before the system
proceeds.

### Step 3: Discover and prepare data

The platform searches approved dataset sources for relevant data. It
assesses whether candidate datasets match the task, language, labels,
domain, and usage terms.

Where appropriate, it can combine suitable public data with
user-provided examples or carefully generated synthetic examples. Data
preparation includes validation, normalization, duplicate detection,
label mapping, and conversion to the format required by the training
pipeline.

The platform should distinguish real examples from synthetic ones,
record dataset provenance, and avoid treating generated examples as
verified ground truth. It should also prevent test data from leaking
into training data.

If there is not enough trustworthy data, SLMForge should explain the
limitation and ask the user for examples or approval to proceed with a
smaller experiment.

### Step 4: Select candidate models

SLMForge searches a curated catalogue of compatible small language
models, potentially drawing from Hugging Face and locally runnable model
ecosystems such as Ollama.

Candidates are compared using relevant factors: - Suitability for the
task and expected output format. - Model size and hardware
requirements. - License and permitted usage. - Supported fine-tuning
methods and tooling. - Expected inference latency and deployment cost. -
Baseline performance on representative examples.

A model being available in a catalogue does not automatically make it
suitable for training or commercial deployment. Compatibility, license,
and resource requirements must be checked.

### Step 5: Establish a baseline

Before training, SLMForge evaluates candidate models using a consistent
validation process. This establishes a reference point and helps
determine whether fine-tuning is likely to provide meaningful
improvement.

Depending on the task, the baseline may use direct inference, a
carefully controlled prompt, or another appropriate non-training
approach.

### Step 6: Plan and run fine-tuning

The system recommends a training strategy based on the task, dataset,
candidate model, compute budget, and expected benefit. For supported
instruction-following tasks, this could include supervised fine-tuning
(SFT) with parameter-efficient methods such as LoRA or QLoRA.

SLMForge generates a reproducible training configuration, checks
resource requirements, and runs the job in an isolated sandbox. Training
progress, logs, resource use, failures, and checkpoints are recorded.

The user should be able to cancel a job and should be informed about
expected compute use before an expensive run begins.

### Step 7: Evaluate and compare

The trained model is evaluated on data that was not used for training.
SLMForge compares it with the baseline using metrics appropriate to the
task.

For a classification task, these may include accuracy, precision,
recall, and F1 score. For structured generation, checks may include
schema validity, exact-match criteria, and task-specific correctness.
Latency, memory use, and approximate serving cost may also be measured.

The platform reports both improvements and regressions. If fine-tuning
does not provide a meaningful benefit, it should recommend keeping the
baseline or trying a different approach rather than claiming success.

### Step 8: Deploy the model

After the user approves the selected result, SLMForge packages the model
or adapter and deploys it using a compatible serving environment.

The user receives: - A hosted API endpoint. - An API key or other
authentication mechanism. - A request and response example. - Model and
version information. - Evaluation results and known limitations. - Basic
usage, latency, and error information.

Deployment should be optional and subject to available infrastructure,
model license, privacy requirements, and cost controls.

## 6. Core Functional Modules

### 6.1 Prompt and Task Analyzer

Interprets the user's request and produces a structured ML
specification. It identifies input and output formats, task type,
constraints, and measurable success criteria. It asks clarifying
questions when the request is ambiguous.

### 6.2 Dataset Discovery and Preparation

Searches approved sources, checks metadata and licensing, validates
data, removes duplicates, maps labels, and converts examples into
supported training formats. It can optionally support synthetic-data
generation and user-uploaded datasets.

### 6.3 Model Discovery and Compatibility

Maintains a catalogue of candidate models and their metadata, including
size, license, supported task types, hardware requirements, and
fine-tuning compatibility. It ranks candidates according to the user's
requirements rather than simply choosing the most popular model.

### 6.4 Training Planner

Chooses whether adaptation is warranted and proposes a suitable
strategy. It can select supported methods such as SFT with LoRA or QLoRA
based on task type, dataset characteristics, hardware limits, and the
desired trade-offs.

### 6.5 Sandboxed Execution Engine

Runs dataset-processing, evaluation, and training jobs in isolated
environments. It manages dependencies, compute limits, logs,
checkpoints, failures, and reproducibility. Uploaded files and generated
scripts must be treated as untrusted inputs.

### 6.6 Evaluation and Model Selection

Runs reproducible evaluations, compares baseline and adapted models,
checks output constraints, and reports quality and efficiency. The
selected model should satisfy minimum acceptance criteria, not merely
have the highest score on a narrow metric.

### 6.7 Deployment and Endpoint Management

Packages the selected model, deploys it to a supported inference
environment, manages endpoint configuration and authentication, and
exposes a documented API. It should support versioning, usage limits,
monitoring, and shutdown of unused deployments.

### 6.8 Project Dashboard

Provides a visible record of each build: the original prompt, parsed
specification, datasets and sources, candidate models, training
configuration, logs, evaluation results, model versions, and endpoint
status.

## 7. High-Level Architecture

``` text
                         USER PROMPT
                              |
                              v
                    Prompt / Task Analyzer
                              |
                              v
                    Structured ML Specification
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Dataset Pipeline           Model Discovery
                 |                         |
                 v                         v
        Validated Dataset             Candidate Models
                 |                         |
                 +------------+------------+
                              |
                              v
                      Baseline Evaluation
                              |
                              v
                        Training Planner
                              |
                              v
                    Isolated Training Sandbox
                              |
                              v
                    Fine-tuning and Checkpoints
                              |
                              v
                    Evaluation and Comparison
                              |
                    +---------+----------+
                    |                    |
                    v                    v
             Meets Criteria?        Needs Revision
                    |                    |
                    v                    v
             User Approval       Adjust Data / Strategy
                    |
                    v
                 Deployment
                    |
                    v
             Hosted Model API
```

This is a logical architecture, not a requirement to implement every
component as a separate microservice. For an early prototype, a modular
backend and a small number of worker processes may be simpler and more
reliable.

## 8. Design Principles

1.  **Prompt-first experience:** Users describe outcomes, not training
    commands.
2.  **Automation with transparency:** The platform automates repetitive
    work while exposing important decisions and evidence.
3.  **Evaluate before training:** Establish a baseline and determine
    whether fine-tuning is justified.
4.  **Data provenance:** Track dataset origins, licenses,
    transformations, and synthetic data.
5.  **Reproducibility:** Store model versions, training configurations,
    random seeds where applicable, and evaluation results.
6.  **Resource awareness:** Estimate compute needs and costs before
    running jobs.
7.  **Safe execution:** Isolate training jobs, validate inputs, and
    restrict credentials and network access.
8.  **User approval:** Require confirmation for costly training and
    deployment actions.
9.  **Honest reporting:** Show uncertainty, limitations, and negative
    results.
10. **Model portability:** Where licenses and formats permit, make
    trained artifacts exportable rather than locking every model into
    one provider.

## 9. What Makes SLMForge Different?

SLMForge's differentiator is the orchestration of the full
model-creation workflow from a natural-language requirement.

-   **Beyond prompt engineering:** It focuses on building a reusable,
    task-specific model when appropriate, not only configuring a prompt
    for a general-purpose model.
-   **Beyond a fine-tuning interface:** Users do not need to arrive with
    a prepared dataset, selected model, and known training method.
-   **End-to-end workflow:** Data discovery, model selection, baseline
    testing, fine-tuning, evaluation, and deployment are connected in
    one process.
-   **Task-aware decisions:** Model and training choices are guided by
    the actual task, constraints, and evaluation criteria.
-   **Evidence-based results:** Users can compare the base model with
    the adapted model and inspect measurable outcomes.
-   **Accessible deployment:** The final output is a usable endpoint,
    not just a checkpoint or training notebook.

These are product goals. The prototype should demonstrate them clearly
rather than claiming that every possible ML task can already be
automated.

## 10. Proposed MVP

The full vision is ambitious. For an ideathon prototype, the first
version should support a narrow set of tasks and a reliable happy path.

### Recommended MVP scope

-   A prompt-based interface for defining a model task.
-   A task analyzer that produces an editable structured specification.
-   A small, curated catalogue of compatible SLMs.
-   Dataset search from a supported source plus optional upload of a CSV
    or JSONL file.
-   Data validation, basic cleaning, and train/validation/test
    splitting.
-   Baseline inference on a selected candidate model.
-   One working fine-tuning path, such as SFT with LoRA or QLoRA, for a
    supported task.
-   A sandboxed training job with progress and logs.
-   A comparison page showing baseline versus fine-tuned results.
-   A demo deployment or local API endpoint for the selected model.

### Suggested demo use case

Use a low-risk, clearly measurable task such as customer-support ticket
classification or converting short support messages into a fixed JSON
schema. This makes it easier to demonstrate data preparation, baseline
evaluation, fine-tuning, and a meaningful before-and-after comparison.

### Defer until later

-   Supporting every model family and ML task.
-   Automatically fine-tuning multiple large models in parallel.
-   Fully autonomous hyperparameter optimization across arbitrary
    hardware.
-   Production-grade multi-region hosting and autoscaling.
-   Complex billing, team management, and enterprise access controls.
-   High-stakes medical, legal, or financial model workflows.

A narrow, functioning pipeline is more convincing than a broad interface
that simulates training without actually doing it.

## 11. Success Metrics

SLMForge should measure both model quality and the quality of the build
process.

### Product metrics

-   Time from prompt submission to a working prototype.
-   Percentage of tasks completed without manual ML configuration.
-   Percentage of builds that finish successfully.
-   Number of user interventions required.
-   Estimated cost per completed build.

### Model metrics

-   Improvement over the baseline on a held-out evaluation set.
-   Task-appropriate quality metrics, such as F1 or exact-match rate.
-   Valid-output rate for constrained formats such as JSON.
-   Inference latency and memory requirements.
-   Performance consistency across relevant input variations.

### Acceptance principle

A build should not be labelled successful solely because training
completed. It should pass task-specific quality checks, satisfy resource
constraints, and clearly disclose limitations. Some tasks may correctly
finish with a recommendation not to fine-tune.

## 12. Technical Considerations

The final stack should be chosen after confirming the available
hardware, budget, and demo environment. Possible implementation
components include:

-   **Frontend:** A web interface for prompts, configuration review,
    build progress, results, and endpoints.
-   **Backend/API:** A service that manages projects, specifications,
    jobs, artifacts, and deployment requests.
-   **Model catalogue:** Metadata from sources such as Hugging Face and
    a curated list of locally runnable models.
-   **Dataset processing:** Python-based utilities for validation,
    transformation, splitting, and deduplication.
-   **Training:** Compatible libraries such as Transformers, PEFT, and
    TRL for supported workflows.
-   **Job execution:** A queue and isolated worker or sandbox for
    long-running tasks.
-   **Artifact storage:** Versioned storage for datasets, checkpoints,
    logs, and evaluation reports.
-   **Inference:** A compatible model-serving runtime with an
    authenticated API.
-   **Monitoring:** Job status, resource usage, failures, latency, and
    endpoint health.

These are candidate technologies, not fixed dependencies. The team
should choose a minimal combination that can be demonstrated reliably.

## 13. Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Insufficient or irrelevant data     Measure dataset fit, report gaps,
                                      request examples, and avoid
                                      unsupported quality claims.

  Synthetic data contains errors      Label synthetic data, validate
                                      samples, and retain held-out real
                                      examples where possible.

  Fine-tuning makes performance worse Compare against a baseline and
                                      retain the option to reject the
                                      trained model.

  Training exceeds available compute  Estimate resource needs, set
                                      limits, use parameter-efficient
                                      methods, and offer smaller
                                      candidates.

  Model or dataset license            Record provenance and check
  restrictions                        permitted use before training or
                                      deployment.

  Malicious uploaded files or scripts Validate uploads and execute jobs
                                      in isolated environments with
                                      restricted permissions.

  Data leakage or privacy exposure    Separate splits, limit retention,
                                      protect secrets, and disclose how
                                      user data is processed.

  Deployment costs grow unexpectedly  Require approval, apply quotas,
                                      monitor usage, and support endpoint
                                      shutdown.

  Misleading evaluation results       Use held-out data, task-specific
                                      metrics, repeatable configurations,
                                      and transparent reports.

  Scope becomes too broad             Start with a small set of supported
                                      tasks and expand based on evidence.
  -----------------------------------------------------------------------

## 14. Privacy, Security, and Responsible Use

Because users may provide private datasets, SLMForge should treat data
handling as a core requirement.

-   Explain where uploaded data is stored and processed.
-   Avoid using customer data to train unrelated models without explicit
    permission.
-   Protect API keys, credentials, and model artifacts.
-   Isolate training jobs from one another and apply CPU, memory, disk,
    GPU, time, and network limits.
-   Validate uploaded archives and data formats before processing.
-   Track the origin and license of datasets and base models.
-   Provide deletion controls for uploaded data and generated artifacts,
    subject to documented retention requirements.
-   Evaluate models for task-specific errors and disclose known
    limitations.
-   Add stronger safeguards and human review for high-impact
    applications.

## 15. Expected Outcome

A successful SLMForge prototype will let a user enter a task
description, review an automatically generated ML specification, prepare
or select data, choose a compatible candidate model, run a supported
fine-tuning workflow, compare the result against a baseline, and access
the selected model through an endpoint.

The main outcome is not simply a trained model. It is a repeatable,
inspectable process that turns an idea into a tested model artifact with
less manual ML setup.

## 16. Long-Term Vision

Over time, SLMForge could expand to more task types, additional model
families, richer dataset discovery, automatic experiment planning,
cost-aware model selection, evaluation suites, private deployment, and
model export.

It could also learn from previous build outcomes to improve its
recommendations, while keeping dataset provenance, reproducibility, and
user control central to the product.

The long-term goal is to make specialized AI models as approachable to
build as small applications: users specify the outcome they want, while
the platform handles the complex workflow needed to produce, validate,
and deliver it.

## 17. Short Pitch

**SLMForge is a prompt-to-SLM platform that automates the journey from
an AI idea to a deployed, task-specific model. Users describe what they
need in natural language, and the platform helps discover data, select a
suitable small language model, choose and run a fine-tuning strategy,
evaluate performance against a baseline, and deploy the result through
an API. By bringing these steps into one transparent workflow, SLMForge
lowers the technical barrier to building specialized language models
while prioritizing measurable results, resource efficiency, and
responsible use.**

------------------------------------------------------------------------

## Appendix: Example Build Specification

**User prompt:**\
"Create a lightweight model that categorizes customer support tickets
into Billing, Technical, Account Access, or General and returns valid
JSON."

**Illustrative specification:** - **Task:** Text classification /
constrained structured output. - **Input:** Ticket text. - **Output:**
JSON object containing the predicted category. - **Allowed labels:**
Billing, Technical, Account Access, General. - **Primary metric:** Macro
F1 or another metric selected for the label distribution. - **Additional
checks:** Valid JSON rate, latency, and memory use. - **Data plan:**
Search for relevant labelled data, then validate or supplement it with
user-provided examples. - **Model plan:** Compare a small number of
compatible candidate models. - **Training plan:** Use a supported
parameter-efficient fine-tuning method if the baseline and data justify
it. - **Evaluation plan:** Compare baseline and adapted models on
held-out examples. - **Deployment output:** Authenticated endpoint with
a request example, response schema, model version, and evaluation
summary.

*This example is illustrative. Actual model choices, scores, training
time, and deployment availability depend on the selected models,
dataset, hardware, and measured results.*
