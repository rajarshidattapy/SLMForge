import { useState } from "react";
import { ApiError, api, type PipelineStage } from "../api";

type Props = {
  projectId: string;
  stages: PipelineStage[];
  nextStage: string | null;
  onChanged: () => void;
};

/** Stages that run on a worker instead of in the request, and their job kinds. */
const LONG_RUNNING: Record<string, string> = {
  baseline: "baseline_eval",
  train: "training",
  evaluate: "evaluation",
};

/** The build's progress, with each stage's availability stated honestly. */
export function Pipeline({ projectId, stages, nextStage, onChanged }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);

  async function run(stage: PipelineStage, approved: boolean) {
    setBusy(stage.name);
    setMessage(null);
    try {
      // Long-running stages never execute in the request: queue them so the
      // flow stays one click per stage — the Jobs panel picks up from here.
      const kind = LONG_RUNNING[stage.name];
      if (kind && stage.implemented) {
        const job = await api.enqueueJob(projectId, kind, {}, approved);
        setMessage({
          tone: "ok",
          text: `Queued ${kind} as job ${job.id.slice(0, 8)} — watch Jobs below.`,
        });
        onChanged();
        return;
      }
      const result = await api.runStage(projectId, stage.name, approved);
      setMessage({
        tone: result.status === "succeeded" ? "ok" : "error",
        text: result.summary || result.error || `Stage ${result.status}.`,
      });
      onChanged();
    } catch (error) {
      // 501 = not built yet, 409 = long-running (use the Dataset/Jobs panels), 428 = approval.
      const detail =
        error instanceof ApiError
          ? error.status === 501
            ? `Not implemented yet — ${error.message}`
            : error.status === 409 && stage.name === "curate_data"
              ? "Building downloads data — approve it in the Dataset panel below (it queues a worker job)."
              : error.message
          : String(error);
      setMessage({ tone: "error", text: detail });
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="panel">
      <header className="panel__head">
        <h2 className="panel__title">Pipeline</h2>
        <span className="muted">Stages marked “not built yet” return 501 by design.</span>
      </header>

      <ol className="pipeline">
        {stages.map((stage, index) => {
          const isNext = stage.name === nextStage;
          return (
            <li
              key={stage.name}
              className={`pipeline__step ${isNext ? "pipeline__step--next" : ""} ${
                stage.implemented ? "" : "pipeline__step--todo"
              }`}
            >
              <span className="pipeline__index" aria-hidden="true">
                {index + 1}
              </span>
              <span className="pipeline__label">{stage.label}</span>
              <div className="pipeline__meta">
                <span className="pipeline__name">{stage.name}</span>
                {!stage.implemented && <span className="badge badge--todo">not built yet</span>}
                {stage.requires_approval && (
                  <span className="badge badge--approval">needs approval</span>
                )}
                {isNext && <span className="badge badge--next">next</span>}
              </div>
              <button
                className="btn btn--small pipeline__action"
                onClick={() => run(stage, stage.requires_approval)}
                disabled={busy !== null}
                title={
                  stage.requires_approval
                    ? "Runs with explicit approval (this action is the approval)"
                    : "Run this stage"
                }
              >
                {busy === stage.name ? "Running…" : "Run"}
              </button>
            </li>
          );
        })}
      </ol>

      {message && (
        <p className={`alert alert--${message.tone === "ok" ? "ok" : "error"}`}>{message.text}</p>
      )}
    </section>
  );
}
