/** Read display metadata out of the project payloads.

 * `Project` carries several opaque JSON columns (spec, dataset_plan,
 * selected_model, training_plan, eval_report). Stages own their shapes, so the
 * dashboard reads them defensively and shows only what is actually present — a
 * missing column renders as "not yet", never as a guess.
 */
import type { Project } from "./api";

export type StatusTone = "good" | "live" | "muted" | "bad";

const STATUS_META: Record<string, { label: string; tone: StatusTone }> = {
  draft: { label: "Draft", tone: "muted" },
  specified: { label: "Specified", tone: "muted" },
  data_ready: { label: "Data ready", tone: "muted" },
  model_selected: { label: "Model selected", tone: "live" },
  baselined: { label: "Baselined", tone: "live" },
  planned: { label: "Plan ready", tone: "live" },
  training: { label: "Training", tone: "live" },
  trained: { label: "Trained", tone: "good" },
  evaluated: { label: "Evaluated", tone: "good" },
  deployed: { label: "Deployed", tone: "good" },
  failed: { label: "Failed", tone: "bad" },
  cancelled: { label: "Cancelled", tone: "muted" },
};

export function statusMeta(status: string): { label: string; tone: StatusTone } {
  return STATUS_META[status] ?? { label: status.replace(/_/g, " "), tone: "muted" };
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function nested(value: unknown, ...path: string[]): unknown {
  let current: unknown = value;
  for (const key of path) {
    const parent = record(current);
    if (!parent) return null;
    current = parent[key];
  }
  return current;
}

/** What kind of ML task the analyzer settled on, if it has run. */
export function taskTypeOf(project: Project): string | null {
  const raw = text(nested(project.spec, "task_type"));
  return raw ? raw.replace(/_/g, " ") : null;
}

/** Which adaptation the planner recommends (finetune / rag / prompt_only). */
export function approachOf(project: Project): string | null {
  return (
    text(nested(project.training_plan, "method")) ??
    text(nested(project.training_plan, "recommendation")) ??
    text(nested(project.spec, "recommended_approach"))
  );
}

/** The base model the build is standing on, from selection or the training plan. */
export function baseModelOf(project: Project): string | null {
  return (
    text(nested(project.selected_model, "selected", "name")) ??
    text(nested(project.selected_model, "selected", "id")) ??
    text(nested(project.training_plan, "base_model_id"))
  );
}

export function primaryMetricOf(project: Project): string | null {
  return text(nested(project.spec, "evaluation", "primary_metric"));
}

/** Names of the datasets the build actually selected, if curation has run. */
export function datasetsOf(project: Project): string[] {
  const plan = record(project.dataset_plan);
  if (!plan) return [];
  const candidates = Array.isArray(plan.candidates) ? plan.candidates : [];
  const selected = Array.isArray(plan.selected) ? plan.selected.map(String) : [];
  return candidates
    .map((candidate) => record(candidate))
    .filter((candidate): candidate is Record<string, unknown> => candidate !== null)
    .filter((candidate) => selected.length === 0 || selected.includes(String(candidate.id)))
    .map((candidate) => text(candidate.name) ?? text(candidate.id))
    .filter((name): name is string => name !== null);
}

/** A one-line "task · model" summary for a build row. */
export function buildSubtitle(project: Project): string {
  const parts = [taskTypeOf(project), approachOf(project), baseModelOf(project)].filter(
    (part): part is string => part !== null,
  );
  return parts.length > 0 ? parts.join(" · ") : "Awaiting analysis";
}

/** "2h ago" style relative time, falling back to a date for old builds. */
export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.floor(minutes)}m ago`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.floor(hours)}h ago`;
  const days = hours / 24;
  if (days < 30) return `${Math.floor(days)}d ago`;
  return new Date(then).toLocaleDateString();
}
