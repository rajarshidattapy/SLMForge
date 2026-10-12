import { useMemo, useState } from "react";
import type { PipelineStage, ProjectDetail as ProjectDetailType } from "../api";
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

/** The build record: what was asked for, what has run, and what was produced. */
export function ProjectDetail({ project, stages, nextStage, onChanged, onDelete }: Props) {
  const [jobsTick, setJobsTick] = useState(0);
  const specJson = useMemo(
    () => (project.spec ? JSON.stringify(project.spec, null, 2) : null),
    [project.spec],
  );
  const spec = (project.spec ?? {}) as Record<string, unknown>;
  const readiness = typeof spec["readiness"] === "string" ? spec["readiness"] : null;
  const generatedBy = typeof spec["generated_by"] === "string" ? spec["generated_by"] : null;
  const labels = Array.isArray(spec["allowed_labels"])
    ? (spec["allowed_labels"] as unknown[]).map(String).join(" · ")
    : null;

  function refreshed() {
    onChanged();
    setJobsTick((n) => n + 1);
  }

  return (
    <div className="detail">
      <header className="detail__head">
        <div>
          <h2 className="detail__title">{project.name}</h2>
          <p className="detail__meta">
            <span className={`status status--${project.status}`}>{project.status}</span>
            <span className="detail__id">{project.id.slice(0, 8)}</span>
            <span>{new Date(project.created_at).toLocaleString()}</span>
          </p>
        </div>
        <button className="btn btn--danger btn--small" onClick={onDelete}>
          Delete
        </button>
      </header>

      <blockquote className="detail__prompt">{project.prompt}</blockquote>

      <Pipeline
        projectId={project.id}
        stages={stages}
        nextStage={nextStage}
        onChanged={refreshed}
      />

      <div className="panel">
        <h2 className="panel__title">Specification</h2>
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
            No specification yet. Run the analyze stage above (and it stays editable before any
            compute is spent).
          </p>
        )}
      </div>

      <DatasetPanel
        projectId={project.id}
        datasetPlan={(project.dataset_plan ?? null) as Record<string, unknown> | null}
        onChanged={refreshed}
      />

      <JobsPanel projectId={project.id} refreshToken={jobsTick} onChanged={onChanged} />

      <div className="panel">
        <h2 className="panel__title">Build log</h2>
        {project.events.length === 0 ? (
          <p className="muted">No events recorded.</p>
        ) : (
          <ul className="events">
            {project.events.map((event) => (
              <li key={event.id} className={`events__item events__item--${event.level}`}>
                <time>{new Date(event.at).toLocaleTimeString()}</time>
                {event.stage && <span className="events__stage">{event.stage}</span>}
                <span className="events__message">{event.message}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="panel">
        <h2 className="panel__title">Artifacts</h2>
        {project.artifacts.length === 0 ? (
          <p className="muted">No artifacts produced yet.</p>
        ) : (
          <ul className="artifacts">
            {project.artifacts.map((artifact) => (
              <li key={artifact.id}>
                <span className="badge">{artifact.kind}</span> {artifact.name}
                <code className="muted"> {artifact.path}</code>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
