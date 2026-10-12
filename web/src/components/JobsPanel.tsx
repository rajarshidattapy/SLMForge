import { useEffect, useRef, useState } from "react";
import { ApiError, api, type Job } from "../api";

type Props = {
  projectId: string;
  refreshToken: number;
  onChanged: () => void;
};

const ACTIVE = new Set(["pending", "running"]);

/** Worker queue: poll while a job is active, cancel on request. */
export function JobsPanel({ projectId, refreshToken, onChanged }: Props) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  // `onChanged` arrives as a fresh closure every parent render; calling it
  // through a ref keeps the polling effect from restarting (and refetching)
  // on each render, which otherwise turns the 5 s poll into a hot loop.
  const onChangedRef = useRef(onChanged);
  onChangedRef.current = onChanged;
  const wasActiveRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function load() {
      try {
        const rows = await api.listJobs(projectId);
        if (cancelled) return;
        setJobs(rows);
        setError(null);
        const active = rows.some((j) => ACTIVE.has(j.status));
        if (active || wasActiveRef.current) {
          // Refresh on every active poll, plus once more when the last active
          // job settles — that final refresh is what reveals the dataset card
          // without the user having to reload the project by hand.
          onChangedRef.current();
        }
        wasActiveRef.current = active;
        if (active) {
          timer = setTimeout(load, 5000);
        }
      } catch (err) {
        if (!cancelled) setError(describe(err));
      }
    }

    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [projectId, refreshToken]);

  async function cancel(jobId: string) {
    try {
      await api.cancelJob(projectId, jobId);
      const rows = await api.listJobs(projectId);
      setJobs(rows);
      onChanged();
    } catch (err) {
      setError(describe(err));
    }
  }

  if (jobs.length === 0 && !error) return null;

  return (
    <div className="panel">
      <h2 className="panel__title">Jobs</h2>
      {error && <p className="alert alert--error">{error}</p>}
      <ul className="pipeline">
        {jobs.map((job) => (
          <li key={job.id} className="pipeline__step">
            <div className="pipeline__main">
              <span className="pipeline__label">
                {job.kind} · {job.status}
              </span>
              <span className="pipeline__name">
                {job.id.slice(0, 8)} · attempts {job.attempts}/{job.max_attempts}
                {job.error ? ` · ${job.error.slice(0, 140)}` : ""}
              </span>
            </div>
            {(job.status === "pending" || job.status === "running") && (
              <button className="btn btn--small btn--danger" onClick={() => void cancel(job.id)}>
                Cancel
              </button>
            )}
          </li>
        ))}
      </ul>
      <p className="muted" style={{ marginTop: 8 }}>
        Long work runs on a worker, never in the request. Start one with `slmforge worker` if a
        job sits at pending.
      </p>
    </div>
  );
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return `API error ${error.status}: ${error.message}`;
  return error instanceof Error ? error.message : String(error);
}
