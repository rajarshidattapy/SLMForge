import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type Capabilities, type PipelineStage, type ProjectDetail } from "./api";
import { CapabilityBar } from "./components/CapabilityBar";
import { ProjectDetail as ProjectDetailView } from "./components/ProjectDetail";
import { PromptForm } from "./components/PromptForm";
import type { Project } from "./api";

export default function App() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [stages, setStages] = useState<PipelineStage[]>([]);
  const [nextStage, setNextStage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshList = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
    } catch (err) {
      setError(describe(err));
    }
  }, []);

  const refreshDetail = useCallback(async (id: string) => {
    try {
      const [project, pipeline, next] = await Promise.all([
        api.getProject(id),
        api.pipeline(id),
        api.nextStep(id),
      ]);
      setDetail(project);
      setStages(pipeline.stages);
      setNextStage(next.next);
    } catch (err) {
      setError(describe(err));
    }
  }, []);

  // Load capabilities and projects once on mount.
  useEffect(() => {
    void (async () => {
      try {
        setCaps(await api.capabilities());
      } catch (err) {
        setError(describe(err));
      }
      await refreshList();
    })();
  }, [refreshList]);

  useEffect(() => {
    if (selectedId) void refreshDetail(selectedId);
    else setDetail(null);
  }, [selectedId, refreshDetail]);

  async function createProject(prompt: string) {
    setBusy(true);
    setError(null);
    try {
      const project = await api.createProject(prompt);
      await refreshList();
      setSelectedId(project.id);
    } catch (err) {
      setError(describe(err));
    } finally {
      setBusy(false);
    }
  }

  async function deleteProject(id: string) {
    setError(null);
    try {
      await api.deleteProject(id);
      if (selectedId === id) setSelectedId(null);
      await refreshList();
    } catch (err) {
      setError(describe(err));
    }
  }

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <span className="brand__mark">◆</span>
          <div>
            <h1 className="brand__name">SLMForge</h1>
            <p className="brand__tag">Describe a model. Review the plan. Ship an endpoint.</p>
          </div>
        </div>
      </header>

      <CapabilityBar caps={caps} />

      <main className="app__main">
        <section className="col col--left">
          <div className="panel">
            <h2 className="panel__title">New build</h2>
            <PromptForm onSubmit={createProject} busy={busy} error={error} />
          </div>

          <div className="panel">
            <h2 className="panel__title">Builds ({projects.length})</h2>
            {projects.length === 0 ? (
              <p className="muted">No builds yet. Start with a requirement above.</p>
            ) : (
              <ul className="project-list">
                {projects.map((project) => (
                  <li key={project.id}>
                    <button
                      className={`project-list__item ${
                        project.id === selectedId ? "project-list__item--active" : ""
                      }`}
                      onClick={() => setSelectedId(project.id)}
                    >
                      <span className="project-list__name">{project.name}</span>
                      <span className={`status status--${project.status}`}>{project.status}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        <section className="col col--right">
          {detail ? (
            <ProjectDetailView
              project={detail}
              stages={stages}
              nextStage={nextStage}
              onChanged={() => {
                if (selectedId) void refreshDetail(selectedId);
                void refreshList();
              }}
              onDelete={() => void deleteProject(detail.id)}
            />
          ) : (
            <div className="panel empty">
              <h2 className="panel__title">No build selected</h2>
              <p className="muted">
                Create a build, or pick one from the list, to see its specification, pipeline
                progress, logs and artifacts.
              </p>
            </div>
          )}
        </section>
      </main>

      <footer className="app__footer">
        <span>SLMForge 0.1.0 — prototype. Stages marked “not built yet” return 501 by design.</span>
      </footer>
    </div>
  );
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return `API error ${error.status}: ${error.message}`;
  return error instanceof Error ? error.message : String(error);
}
