import { useState } from "react";
import { ApiError, api } from "../api";

type License = {
  id?: string;
  commercial_ok?: boolean | null;
  unknown?: boolean;
};

type Candidate = {
  id: string;
  name: string;
  parameters_b?: number | null;
  context_window?: number | null;
  license?: License | null;
  local_available?: boolean;
  downloaded?: boolean;
  instruction_tuned?: boolean;
  rank_score?: number | null;
  rank_rationale?: string | null;
  warnings?: string[];
  finetune?: { min_vram_gb?: number | null; methods?: string[] } | null;
};

type Props = {
  projectId: string;
  selection: Record<string, unknown> | null;
  onChanged: () => void;
};

function asShortlist(selection: Record<string, unknown> | null): Candidate[] {
  if (!selection) return [];
  const raw = selection["shortlist"];
  return Array.isArray(raw) ? (raw as Candidate[]) : [];
}

function asSelected(selection: Record<string, unknown> | null): Candidate | null {
  if (!selection) return null;
  const raw = selection["selected"];
  if (!raw || typeof raw !== "object") return null;
  return raw as Candidate;
}

/** Base-model review: rank, approve one, or reject with notes. */
export function ModelPanel({ projectId, selection, onChanged }: Props) {
  const shortlist = asShortlist(selection);
  const selected = asSelected(selection);
  const comparedOn = Array.isArray(selection?.["compared_on"])
    ? (selection["compared_on"] as unknown[]).map(String)
    : [];
  const decisionNotes =
    typeof selection?.["decision_notes"] === "string" ? selection["decision_notes"] : null;

  const [picked, setPicked] = useState<string>("");
  const [notes, setNotes] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);

  const effectivePicked = picked || (shortlist[0]?.id ?? "");

  async function rank(offline: boolean) {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.runStage(projectId, "select_model", true, offline ? { offline: true } : {});
      setMessage({ tone: result.status === "succeeded" ? "ok" : "error", text: result.summary });
      onChanged();
    } catch (error) {
      setMessage({ tone: "error", text: describe(error) });
    } finally {
      setBusy(false);
    }
  }

  async function approve() {
    if (!effectivePicked) {
      setMessage({ tone: "error", text: "Pick a candidate first." });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.runStage(projectId, "select_model", true, {
        selected: effectivePicked,
        ...(notes.trim() ? { decision_notes: notes.trim() } : {}),
      });
      setMessage({ tone: result.status === "succeeded" ? "ok" : "error", text: result.summary });
      onChanged();
    } catch (error) {
      setMessage({ tone: "error", text: describe(error) });
    } finally {
      setBusy(false);
    }
  }

  async function reject() {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.runStage(projectId, "select_model", true, {
        rejected: true,
        ...(notes.trim() ? { decision_notes: notes.trim() } : {}),
      });
      setMessage({ tone: result.status === "succeeded" ? "ok" : "error", text: result.summary });
      onChanged();
    } catch (error) {
      setMessage({ tone: "error", text: describe(error) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2 className="panel__title">Base model</h2>

      {selected ? (
        <p className="muted" style={{ marginBottom: 8 }}>
          Selected: <strong>{selected.name}</strong>
          {selected.parameters_b != null && ` (${selected.parameters_b}B)`}
          {selected.rank_score != null && ` · score ${selected.rank_score}`}
          {decisionNotes ? ` — ${decisionNotes}` : ""}
        </p>
      ) : (
        <p className="muted" style={{ marginBottom: 8 }}>
          No approved model yet. Rank the candidates, pick one, and approve it —
          or reject the recommendation with notes.
        </p>
      )}

      <div className="prompt__actions" style={{ marginBottom: 12 }}>
        <button className="btn btn--small" onClick={() => void rank(false)} disabled={busy}>
          {busy ? "Working…" : shortlist.length === 0 ? "Rank models" : "Re-rank models"}
        </button>
        <button className="btn btn--small btn--ghost" onClick={() => void rank(true)} disabled={busy} title="Skip the Hub search; rank the hand-verified catalog only.">
          Rank offline
        </button>
        <span className="muted">Ranking is read-only and needs nothing downloaded.</span>
      </div>

      {shortlist.length > 0 && (
        <>
          {comparedOn.length > 0 && (
            <p className="muted" style={{ marginBottom: 8 }}>
              Compared on: {comparedOn.join(" · ")}
            </p>
          )}
          <ul className="pipeline">
            {shortlist.map((c) => (
              <li key={c.id} className="pipeline__step">
                <label style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                  <input
                    type="radio"
                    name="model-candidate"
                    checked={effectivePicked === c.id}
                    onChange={() => setPicked(c.id)}
                    disabled={busy}
                  />
                  <span className="pipeline__main">
                    <span className="pipeline__label">
                      {c.name} {c.rank_score != null && `(score ${c.rank_score})`}
                    </span>
                    <span className="pipeline__name">{c.id}</span>
                    <span className="muted">
                      {[
                        c.parameters_b != null ? `${c.parameters_b}B params` : null,
                        c.finetune?.min_vram_gb != null
                          ? `QLoRA ~${c.finetune.min_vram_gb} GB`
                          : null,
                        c.context_window != null ? `ctx ${c.context_window}` : null,
                        c.license?.id ? `licence ${c.license.id}` : "licence unknown",
                        c.downloaded ? "downloaded" : c.local_available ? "via Ollama" : "Hub download at train time",
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                    {c.rank_rationale && <span className="muted">{c.rank_rationale}</span>}
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
            <input
              aria-label="Decision notes"
              placeholder="Decision notes (optional)"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              disabled={busy}
              style={{ flex: 1, minWidth: 180 }}
            />
            <button className="btn btn--primary btn--small" onClick={() => void approve()} disabled={busy}>
              {busy ? "Working…" : "Approve"}
            </button>
            <button className="btn btn--small btn--danger" onClick={() => void reject()} disabled={busy}>
              Reject
            </button>
          </div>
        </>
      )}

      {message && <p className={`alert alert--${message.tone === "ok" ? "ok" : "error"}`}>{message.text}</p>}
    </div>
  );
}

function describe(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 428) return `Approval required: ${error.message}`;
    if (error.status === 501) return `Not built yet — ${error.message}`;
    return `API error ${error.status}: ${error.message}`;
  }
  return error instanceof Error ? error.message : String(error);
}
