import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  api,
  type AuthStatus,
  type Capabilities,
  type PipelineStage,
  type Project,
  type ProjectDetail,
} from "./api";
import { BuildList } from "./components/BuildList";
import { DocsView } from "./components/DocsView";
import { EmptyBuild } from "./components/EmptyBuild";
import { HeroArt } from "./components/HeroArt";
import { InspirationPanel } from "./components/InspirationPanel";
import { ProjectDetail as ProjectDetailView } from "./components/ProjectDetail";
import { PromptForm } from "./components/PromptForm";
import { Sidebar, type View } from "./components/Sidebar";
import { TemplatesView } from "./components/TemplatesView";
import { BoltIcon, GridIcon, ImportIcon } from "./icons";
import { IMPORT_EXAMPLE_PROMPT, TICKET_CLASSIFIER_PROMPT, type Template } from "./templates";

export default function App() {
  const [view, setView] = useState<View>("home");
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [stages, setStages] = useState<PipelineStage[]>([]);
  const [nextStage, setNextStage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshAuth = useCallback(async () => {
    try {
      setAuth(await api.authStatus());
    } catch (err) {
      setError(describe(err));
    }
  }, []);

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

  // Load capabilities, sign-in state and projects once on mount.
  useEffect(() => {
    // The OAuth callback bounces back here with ?auth_error=… on failure; surface
    // it once, then drop it from the URL so a refresh does not resurrect it.
    const params = new URLSearchParams(window.location.search);
    const authError = params.get("auth_error");
    if (authError) {
      setError(`Hugging Face sign-in failed (${authError}).`);
      params.delete("auth_error");
      const query = params.toString();
      window.history.replaceState({}, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
    }
    void (async () => {
      try {
        setCaps(await api.capabilities());
      } catch (err) {
        setError(describe(err));
      }
      await Promise.all([refreshAuth(), refreshList()]);
    })();
  }, [refreshAuth, refreshList]);

  useEffect(() => {
    if (selectedId) void refreshDetail(selectedId);
    else setDetail(null);
  }, [selectedId, refreshDetail]);

  async function createProject(text: string, name: string | null) {
    setBusy(true);
    setError(null);
    try {
      const project = await api.createProject(text, name ?? undefined);
      await refreshList();
      setPrompt("");
      setSelectedId(project.id);
    } catch (err) {
      setError(describe(err));
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    setError(null);
    try {
      await api.logout();
    } catch (err) {
      setError(describe(err));
    }
    await refreshAuth();
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

  /** A template fills the composer; nothing is built until the user submits it. */
  function useTemplate(template: Template) {
    setPrompt(template.prompt);
    setView("home");
  }

  const selected = detail ? (
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
    <EmptyBuild />
  );

  const workbench = (listTitle: string, limit?: number) => (
    <>
      <div className="workbench">
        <div className="workbench__col">
          <BuildList
            title={listTitle}
            limit={limit}
            projects={projects}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onDelete={(id) => void deleteProject(id)}
            onViewAll={limit === undefined ? undefined : () => setView("builds")}
          />
        </div>
        <aside className="workbench__col workbench__col--side">{selected}</aside>
      </div>
      {limit !== undefined && (
        <InspirationPanel onBrowse={() => setView("templates")} onPick={useTemplate} />
      )}
    </>
  );

  return (
    <div className="shell">
      <Sidebar
        view={view}
        onNavigate={setView}
        caps={caps}
        auth={auth}
        onSignOut={() => void signOut()}
        buildCount={projects.length}
      />

      <main className="main">
        {error && <p className="alert alert--error">{error}</p>}

        {view === "home" && (
          <div className="home">
            <section className="hero">
              <HeroArt />
              <div className="hero__copy">
                <h1 className="hero__title">
                  Turn your idea into <span className="hero__accent">a production-ready SLM</span>
                </h1>
                <p className="hero__sub">
                  Describe a model. We’ll generate a build plan, including training, evaluation and
                  deployment steps.
                </p>
              </div>

              <div className="hero__starters">
                <button
                  type="button"
                  className={`chip-btn ${
                    prompt === TICKET_CLASSIFIER_PROMPT ? "chip-btn--on" : ""
                  }`}
                  onClick={() => setPrompt(TICKET_CLASSIFIER_PROMPT)}
                >
                  <BoltIcon size={16} /> Quick Start
                </button>
                <button type="button" className="chip-btn" onClick={() => setView("templates")}>
                  <GridIcon size={16} /> Use template
                </button>
                <button
                  type="button"
                  className="chip-btn"
                  onClick={() => setPrompt(IMPORT_EXAMPLE_PROMPT)}
                >
                  <ImportIcon size={16} /> Import from example
                </button>
              </div>

              <PromptForm
                prompt={prompt}
                onPromptChange={setPrompt}
                onSubmit={createProject}
                busy={busy}
              />
            </section>

            {workbench("Recent builds", 4)}
          </div>
        )}

        {view === "builds" && (
          <section className="page">
            <header className="page__head">
              <h1 className="page__title">Builds</h1>
              <p className="page__sub">
                {projects.length === 0
                  ? "No builds yet — describe a model from the home screen to start one."
                  : `${projects.length} build${projects.length === 1 ? "" : "s"} in this workspace.`}
              </p>
            </header>
            {workbench("All builds")}
          </section>
        )}

        {view === "templates" && <TemplatesView onUse={useTemplate} />}
        {view === "docs" && <DocsView />}
      </main>
    </div>
  );
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return `API error ${error.status}: ${error.message}`;
  return error instanceof Error ? error.message : String(error);
}
