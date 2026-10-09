import { useState } from "react";
import { ApiError, api, type PipelineStage } from "../api";

type Props = {
  projectId: string;
  stages: PipelineStage[];
  nextStage: string | null;
  onChanged: () => void;
};

/** The build's progress, with each stage's availability stated honestly. */
export function Pipeline({ projectId, stages, nextStage, onChanged }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);

  async function run(stage: PipelineStage, approved: boolean) {
    setBusy(stage.name);
    setMessage(null);
    try {
      const result = await api.runStage(projectId, stage.name, approved);
      setMessage({
        tone: result.status === "succeeded" ? "ok" : "error",
        text: result.summary || result.error || `Stage ${result.status}.`,
      });
      onChanged();
    } catch (error) {
      // 501 = not built yet, 409 = long-running, 428 = approval required.
      const detail =
        error instanceof ApiError
          ? error.status === 501
            ? `Not implemented yet — ${error.message}`
            : error.message
          : String(error);
      setMessage({ tone: "error", text: detail });
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="panel">
      <h2 className="panel__title">Pipeline</h2>
      <ol className="pipeline">
        {stages.map((stage) => {
          const isNext = stage.name === nextStage;
          return (
            <li
              key={stage.name}
              className={`pipeline__step ${isNext ? "pipeline__step--next" : ""} ${
                stage.implemented ? "" : "pipeline__step--todo"
              }`}
            >
              <div className="pipeline__main">
                <span className="pipeline__label">{stage.label}</span>
                <span className="pipeline__name">{stage.name}</span>
              </div>
              <div className="pipeline__badges">
                {!stage.implemented && <span className="badge badge--todo">not built yet</span>}
                {stage.requires_approval && (
                  <span className="badge badge--approval">needs approval</span>
                )}
                {isNext && <span className="badge badge--next">next</span>}
              </div>
              <button
                className="btn btn--small"
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
    </div>
  );
}
