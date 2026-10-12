import { useState } from "react";
import { ApiError, api } from "../api";

type Candidate = {
  id: string;
  name: string;
  fit_score: number | null;
  fit_rationale?: string | null;
  rows?: number | null;
  label_coverage?: Record<string, number>;
  warnings?: string[];
  license_note?: string | null;
  commercial_ok?: boolean | null;
};

type DatasetCard = {
  total_examples: number;
  splits: { train: number; validation: number; test: number };
  label_distribution: Record<string, number>;
  synthetic_fraction: number;
  sufficiency?: string;
  issues: { kind: string; severity: string; count: number; detail?: string | null }[];
  sources: { source?: string; uri?: string; license?: { id?: string } | string | null }[];
};

type Props = {
  projectId: string;
  datasetPlan: Record<string, unknown> | null;
  onChanged: () => void;
};

function asCandidates(plan: Record<string, unknown> | null): Candidate[] {
  if (!plan) return [];
  const raw = plan["candidates"];
  return Array.isArray(raw) ? (raw as Candidate[]) : [];
}

function asCard(plan: Record<string, unknown> | null): DatasetCard | null {
  if (!plan) return null;
  const raw = plan["card"];
  if (!raw || typeof raw !== "object") return null;
  return raw as DatasetCard;
}

/** Dataset review: plan candidates, approve-and-build via the worker, and card. */
export function DatasetPanel({ projectId, datasetPlan, onChanged }: Props) {
  const candidates = asCandidates(datasetPlan);
  const card = asCard(datasetPlan);
  const sufficiency = typeof datasetPlan?.["sufficiency"] === "string" ? datasetPlan["sufficiency"] : null;
  const [selected, setSelected] = useState<string>("");
  const [target, setTarget] = useState<number>(4000);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);

  const effectiveSelected = selected || (candidates[0]?.id ?? "");

  async function plan() {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.runStage(projectId, "curate_data");
      setMessage({ tone: result.status === "succeeded" ? "ok" : "error", text: result.summary });
      onChanged();
    } catch (error) {
      setMessage({ tone: "error", text: describe(error) });
    } finally {
      setBusy(false);
    }
  }

  async function build() {
    if (!effectiveSelected) {
      setMessage({ tone: "error", text: "Pick a dataset candidate first." });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const job = await api.enqueueJob(
        projectId,
        "dataset_build",
        { build: true, selected: [effectiveSelected], target_examples: target },
        true,
      );
      setMessage({ tone: "ok", text: `Build queued as job ${job.id.slice(0, 8)} — watch Jobs below.` });
      onChanged();
    } catch (error) {
      setMessage({ tone: "error", text: describe(error) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2 className="panel__title">Dataset</h2>
      <div className="prompt__actions" style={{ marginBottom: 12 }}>
        <button className="btn btn--small" onClick={plan} disabled={busy}>
          {busy ? "Working…" : candidates.length === 0 ? "Plan dataset" : "Re-plan dataset"}
        </button>
        <span className="muted">
          Planning profiles candidates remotely and downloads nothing. Building downloads, so it
          runs on a worker.
        </span>
      </div>

      {candidates.length > 0 && (
        <>
          <p className="muted" style={{ marginBottom: 8 }}>
            {String(datasetPlan?.["sufficiency_notes"] ?? "") ||
              (sufficiency ? `Sufficiency: ${sufficiency}.` : "")}{" "}
            Pick a candidate, approve the count, then queue the build.
          </p>
          <ul className="pipeline">
            {candidates.map((c) => (
              <li key={c.id} className="pipeline__step">
                <label style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                  <input
                    type="radio"
                    name="dataset-candidate"
                    checked={effectiveSelected === c.id}
                    onChange={() => setSelected(c.id)}
                  />
                  <span className="pipeline__main">
                    <span className="pipeline__label">
                      {c.name} {c.fit_score != null && `(fit ${c.fit_score})`}
                    </span>
                    <span className="pipeline__name">{c.id}</span>
                    {c.fit_rationale && <span className="muted">{c.fit_rationale}</span>}
                    {c.rows != null && <span className="muted">rows: {c.rows}</span>}
                    {c.label_coverage && (
                      <span className="muted">
                        coverage:{" "}
                        {Object.entries(c.label_coverage)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join(" · ")}
                      </span>
                    )}
                    {c.license_note && <span className="muted">licence: {c.license_note}</span>}
                    {(c.warnings ?? []).map((w) => (
                      <span key={w} className="muted" style={{ color: "var(--warn)" }}>
                        ! {w}
                      </span>
                    ))}
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <div className="prompt__actions" style={{ marginTop: 12 }}>
            <label className="muted" htmlFor="target-examples">
              Target examples
            </label>
            <input
              id="target-examples"
              type="number"
              min={100}
              max={20000}
              step={100}
              value={target}
              onChange={(e) => setTarget(Number(e.target.value) || 4000)}
              style={{ width: 110 }}
            />
            <button className="btn btn--primary btn--small" onClick={build} disabled={busy}>
              {busy ? "Queueing…" : "Approve & build"}
            </button>
          </div>
        </>
      )}

      {card && (
        <div style={{ marginTop: 12 }}>
          <h3 className="panel__title">Dataset card</h3>
          <p className="muted">
            {card.total_examples} examples — train {card.splits.train} / validation{" "}
            {card.splits.validation} / test {card.splits.test} · synthetic{" "}
            {Math.round(card.synthetic_fraction * 100)}%
            {sufficiency ? ` · ${sufficiency}` : ""}
          </p>
          <p className="muted">
            labels:{" "}
            {Object.entries(card.label_distribution)
              .map(([k, v]) => `${k}: ${v}`)
              .join(" · ")}
          </p>
          {card.sources.map((s, i) => (
            <p key={i} className="muted">
              source: {s.source ?? "?"} {s.uri ?? ""} licence:{" "}
              {typeof s.license === "string" ? s.license : (s.license?.id ?? "?")}
            </p>
          ))}
          {card.issues.map((issue, i) => (
            <p key={i} className="muted" style={{ color: issue.severity === "error" ? "var(--bad)" : undefined }}>
              [{issue.severity}] {issue.kind}: {issue.count}
              {issue.detail ? ` — ${issue.detail}` : ""}
            </p>
          ))}
        </div>
      )}

      {message && <p className={`alert alert--${message.tone === "ok" ? "ok" : "error"}`}>{message.text}</p>}
    </div>
  );
}

function describe(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) return `Long-running work: ${error.message}`;
    if (error.status === 428) return `Approval required: ${error.message}`;
    if (error.status === 501) return `Not built yet — ${error.message}`;
    return `API error ${error.status}: ${error.message}`;
  }
  return error instanceof Error ? error.message : String(error);
}
