/** Typed client for the SLMForge API.
 *
 * Relative URLs, so the same code works through the Vite dev proxy and when the
 * FastAPI app serves the built assets itself.
 */

const BASE = "/api";

export type Capabilities = {
  python_version: string;
  platform: string;
  sandbox_backend: string;
  gpu_available: boolean;
  gpu_name: string | null;
  gpu_vram_gb: number | null;
  torch_available: boolean;
  torch_device: string | null;
  ollama_available: boolean;
  ollama_models: string[];
  planner_llm_configured: boolean;
  planner_llm_model: string | null;
  hf_token_configured: boolean;
  hf_push_enabled: boolean;
  artifact_root: string;
  warnings: string[];
};

export type Project = {
  id: string;
  name: string;
  prompt: string;
  status: string;
  spec: Record<string, unknown> | null;
  dataset_plan: Record<string, unknown> | null;
  selected_model: Record<string, unknown> | null;
  training_plan: Record<string, unknown> | null;
  eval_report: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

export type ProjectDetail = Project & {
  events: EventItem[];
  artifacts: Artifact[];
};

export type EventItem = {
  id: string;
  at: string;
  level: string;
  stage: string | null;
  message: string;
  data: Record<string, unknown> | null;
};

export type Artifact = {
  id: string;
  kind: string;
  name: string;
  path: string;
  size_bytes: number | null;
  sha256: string | null;
  meta: Record<string, unknown> | null;
  created_at: string;
};

export type PipelineStage = {
  name: string;
  label: string;
  requires_approval: boolean;
  implemented: boolean;
};

export type NextStep = {
  next: string | null;
  label: string;
  requires_approval: boolean;
};

export type StageRunResult = {
  stage: string;
  status: string;
  summary: string;
  duration_s: number | null;
  metrics: Record<string, number>;
  blocked_on: string[];
  needs_approval: boolean;
  error: string | null;
  output: Record<string, unknown>;
};

/** Error carrying the HTTP status so callers can explain 409/428/501 precisely. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // Response had no JSON body; the status line is the best we have.
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  capabilities: () => request<Capabilities>("/capabilities"),

  listProjects: () => request<Project[]>("/projects"),

  createProject: (prompt: string, name?: string) =>
    request<Project>("/projects", {
      method: "POST",
      body: JSON.stringify({ prompt, name: name || null }),
    }),

  getProject: (id: string) => request<ProjectDetail>(`/projects/${id}`),

  deleteProject: (id: string) =>
    request<void>(`/projects/${id}`, { method: "DELETE" }),

  pipeline: (id: string) => request<{ stages: PipelineStage[] }>(`/projects/${id}/pipeline`),

  nextStep: (id: string) => request<NextStep>(`/projects/${id}/next`),

  saveSpec: (id: string, spec: Record<string, unknown>) =>
    request<{ spec: Record<string, unknown>; status: string }>(`/projects/${id}/spec`, {
      method: "PUT",
      body: JSON.stringify({ spec }),
    }),

  runStage: (id: string, stage: string, approved = false) =>
    request<StageRunResult>(`/projects/${id}/stages/${stage}/run`, {
      method: "POST",
      body: JSON.stringify({ approved }),
    }),
};
