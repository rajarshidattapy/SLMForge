
import { useMemo, useState } from "react";

import type { PipelineStage, ProjectDetail as ProjectDetailType } from "../api";
import { FilePlanIcon, TrashIcon } from "../icons";
import {
  approachOf,
  baseModelOf,
  datasetsOf,
  primaryMetricOf,
  statusMeta,
  taskTypeOf,
} from "../projectMeta";
import { DatasetPanel } from "./DatasetPanel";
import { JobsPanel } from "./JobsPanel";
import { Pipeline } from "./Pipeline";

type Props = {
  project: ProjectDetailType;
  stages: PipelineStage[];
  nextStage: string | null;
  onChanged: () => void;
  onDelete: () => void;
};

type Tab = "spec" | "logs" | "artifacts";

/** The build record: what was asked for, what has run, and what was produced. */
export function ProjectDetail({
  project,
  stages,
  nextStage,
  onChanged,
  onDelete,
}: Props) {
  const [jobsTick, setJobsTick] = useState(0);
  const [tab, setTab] = useState<Tab>("spec");
  const status = statusMeta(project.status);

  const specJson = useMemo(
    () => (project.spec ? JSON.stringify(project.spec, null, 2) : null),
    [project.spec],
  );

  const spec = (project.spec ?? {}) as Record<string, unknown>;

  const readiness =
    typeof spec["readiness"] === "string" ? spec["readiness"] : null;
  const generatedBy =
    typeof spec["generated_by"] === "string" ? spec["generated_by"] : null;

  const labels = Array.isArray(spec["allowed_labels"])
    ? (spec["allowed_labels"] as unknown[]).map(String).join(" · ")
    : null;

  function refreshed() {
    onChanged();
    setJobsTick((n) => n + 1);
  }

  const facts = [
    { label: "Task", value: taskTypeOf(project) },
    { label: "Approach", value: approachOf(project) },
    { label: "Base model", value: baseModelOf(project) },
    { label: "Metric", value: primaryMetricOf(project) },
    { label: "Data", value: datasetsOf(project).join(", ") || null },
  ].filter(
    (fact): fact is { label: string; value: string } =>
      fact.value !== null,
  );

  return (
    <div className="detail">
      <header className="detail__head">
        <span className="detail__glyph" aria-hidden="true">
          <FilePlanIcon size={22} />
        </span>

        <div className="detail__heading">
          <h2 className="detail__title">{project.name}</h2>
          <p className="detail__meta">
            <span className={`pill pill--${status.tone}`}>
              {status.label}
            </span>
            <span className="detail__id">{project.id.slice(0, 8)}</span>
            <span>{new Date(project.created_at).toLocaleString()}</span>
          </p>
        </div>

        <button
          className="icon-btn icon-btn--ghost"
          onClick={onDelete}
          aria-label="Delete build"
        >
          <TrashIcon size={18} />
        </button>
      </header>

      {facts.length > 0 && (
        <dl className="facts">
          {facts.map((fact) => (
            <div key={fact.label} className="fact">
              <dt>{fact.label}</dt>
              <dd>{fact.value}</dd>
            </div>
          ))}
        </dl>
      )}

      <blockquote className="detail__prompt">{project.prompt}</blockquote>

      <Pipeline
        projectId={project.id}
        stages={stages}
        nextStage={nextStage}
        onChanged={refreshed}
      />

      <section className="panel">
        <div className="tabs" role="tablist">
          <TabButton id="spec" current={tab} onSelect={setTab} label="Specification" />
          <TabButton
            id="logs"
            current={tab}
            onSelect={setTab}
            label={`Build log${project.events.length ? ` (${project.events.length})` : ""}`}
          />
          <TabButton
            id="artifacts"
            current={tab}
            onSelect={setTab}
            label={`Artifacts${project.artifacts.length ? ` (${project.artifacts.length})` : ""}`}
          />
        </div>

        {tab === "spec" && (
          <>
            {specJson ? (
              <>
                <p className="muted" style={{ marginBottom: 8 }}>
                  {labels ?? "No labels yet"}
                  {readiness ? ` · ${readiness}` : ""}
                  {generatedBy ? ` · via ${generatedBy}` : ""}
                </p>
                <pre className="code">{specJson}</pre>
              </>
            ) : (
              <p className="muted">
                No specification yet. The analyze stage generates one (and it stays editable before any compute is spent).
              </p>
            )}

            <DatasetPanel
              // Reset selection/target per project: the panel keeps local state
              // that is meaningless for a different project's candidates.
              key={project.id}
              projectId={project.id}
              datasetPlan={
                (project.dataset_plan ?? null) as
                  | Record<string, unknown>
                  | null
              }
              onChanged={refreshed}
            />

            <JobsPanel
              projectId={project.id}
              refreshToken={jobsTick}
              onChanged={onChanged}
            />
          </>
        )}

        {tab === "logs" &&
          (project.events.length === 0 ? (
            <p className="muted">No events recorded.</p>
          ) : (
            <ul className="events">
              {project.events.map((event) => (
                <li
                  key={event.id}
                  className={`events__item events__item--${event.level}`}
                >
                  <time>{new Date(event.at).toLocaleTimeString()}</time>
                  {event.stage && (
                    <span className="events__stage">{event.stage}</span>
                  )}
                  <span className="events__message">{event.message}</span>
                </li>
              ))}
            </ul>
          ))}

        {tab === "artifacts" &&
          (project.artifacts.length === 0 ? (
            <p className="muted">No artifacts produced yet.</p>
          ) : (
            <ul className="artifacts">
              {project.artifacts.map((artifact) => (
                <li key={artifact.id}>
                  <span className="badge">{artifact.kind}</span>{" "}
                  {artifact.name}
                  <code className="muted"> {artifact.path}</code>
                </li>
              ))}
            </ul>
          ))}
      </section>
    </div>
  );
}

function TabButton({
  id,
  current,
  onSelect,
  label,
}: {
  id: Tab;
  current: Tab;
  onSelect: (tab: Tab) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={current === id}
      className={`tab ${current === id ? "tab--active" : ""}`}
      onClick={() => onSelect(id)}
    >
      {label}
    </button>
  );
}