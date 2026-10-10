import { CheckIcon } from "../icons";

/** Order and purpose of the eight stages, mirroring the backend pipeline. */
const STAGES: { name: string; label: string; purpose: string }[] = [
  {
    name: "analyze",
    label: "Analyze requirement",
    purpose: "Turn the prompt into an editable specification: task, I/O contract, metrics, risks.",
  },
  {
    name: "curate_data",
    label: "Curate dataset",
    purpose: "Find or prepare training data, check licences, dedupe, split, and record provenance.",
  },
  {
    name: "select_model",
    label: "Select base model",
    purpose: "Compare compatible small models on task fit, licence, size and hardware.",
  },
  {
    name: "baseline",
    label: "Establish baseline",
    purpose: "Measure the untouched model first, so any claimed improvement is real.",
  },
  {
    name: "plan_training",
    label: "Plan fine-tuning",
    purpose: "Choose SFT/LoRA/QLoRA or recommend against training, with a cost estimate.",
  },
  {
    name: "train",
    label: "Fine-tune",
    purpose: "Run the adapter training inside the sandbox; log steps, loss and checkpoints.",
  },
  {
    name: "evaluate",
    label: "Evaluate & compare",
    purpose: "Score baseline vs adapted on held-out data, including failures and regressions.",
  },
  {
    name: "deploy",
    label: "Deploy endpoint",
    purpose: "Package the chosen model behind an authenticated API, with limitations attached.",
  },
];

export function DocsView() {
  return (
    <section className="page">
      <header className="page__head">
        <h1 className="page__title">Docs</h1>
        <p className="page__sub">
          How SLMForge builds a model, and the rules the pipeline will not break to look good.
        </p>
      </header>

      <div className="doc">
        <h2 className="doc__title">Pipeline stages</h2>
        <ol className="doc__stages">
          {STAGES.map((stage) => (
            <li key={stage.name} className="doc__stage">
              <h3 className="doc__stage-name">{stage.label}</h3>
              <p className="muted">{stage.purpose}</p>
            </li>
          ))}
        </ol>
      </div>

      <div className="doc">
        <h2 className="doc__title">Two rules</h2>
        <ul className="doc__rules">
          <li>
            <CheckIcon size={16} />
            <span>
              <strong>Evidence over enthusiasm.</strong> A build is not successful because training
              exited 0. It is successful when the adapted model beats the baseline on held-out data
              — or when the system honestly says not to fine-tune.
            </span>
          </li>
          <li>
            <CheckIcon size={16} />
            <span>
              <strong>Transparency and consent.</strong> Costly or irreversible steps (training,
              deployment) require an approval gate, a pre-flight estimate and a plain-English record
              of what was chosen and why.
            </span>
          </li>
        </ul>
      </div>

      <div className="doc">
        <h2 className="doc__title">Deeper reading</h2>
        <ul className="doc__links">
          <li>
            <code>docs/ARCHITECTURE.md</code> — data flow, stage contracts, sandbox and security
            model.
          </li>
          <li>
            <code>docs/ROADMAP.md</code> — what is implemented, what is next, and the exit criteria.
          </li>
          <li>
            <code>docs/AGENTS.md</code> — repository conventions and environment notes.
          </li>
        </ul>
      </div>
    </section>
  );
}
