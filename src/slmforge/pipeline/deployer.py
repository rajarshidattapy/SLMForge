"""Stage 9 — Deployment & Endpoint Management.

Publishes the selected model and returns a callable endpoint with versioning,
auth and usage reporting.

Planned algorithm
-----------------
1. Merge the adapter into the base model (when requested) and record the version.
2. Push the model/adapter to the HF Hub (`settings.hf_org`, private by default)
   with a model card that includes the eval summary and known limitations.
3. Create the endpoint per `settings.deploy_target`:
   - `hf_endpoint`: HF Inference Endpoint on the chosen instance type;
   - `hf_space`: a Gradio/FastAPI Space for demo traffic;
   - `local_openai`: an OpenAI-compatible server on the local GPU.
4. Provision an API key, store only a secret reference, and return a request +
   response example.
5. Record usage/latency/errors and support shutdown of unused deployments.
6. Require explicit user approval; never auto-deploy.

Exit criteria: a `DeployRecord` whose endpoint has been called successfully once,
with the key stored as a reference and cost/usage visible.
"""

from __future__ import annotations

from ..models import StageName
from .base import Stage, StageContext, StageResult


class DeployerStage(Stage):
    name = StageName.deploy
    label = "Deploy endpoint"
    requires_approval = True
    implemented = False  # Stage 8

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError(
            "DeployerStage: implement HF Hub push, endpoint creation and usage "
            "reporting (see docs/ROADMAP.md Stage 8)."
        )
