import type { Capabilities } from "../api";

type Props = { caps: Capabilities | null };

/** Shows what this machine can actually do, and warns before a build fails later. */
export function CapabilityBar({ caps }: Props) {
  if (!caps) return <div className="caps caps--loading">Checking environment…</div>;

  const gpu = caps.gpu_available
    ? `${caps.gpu_name ?? "GPU"} · ${caps.gpu_vram_gb ?? "?"} GB`
    : "CPU only";

  return (
    <section className="caps">
      <div className="caps__row">
        <Chip label="Compute" value={gpu} tone={caps.gpu_available ? "good" : "warn"} />
        <Chip
          label="Sandbox"
          value={caps.sandbox_backend}
          tone={caps.sandbox_backend === "docker" ? "good" : "warn"}
        />
        <Chip
          label="Ollama"
          value={caps.ollama_available ? `${caps.ollama_models.length} models` : "offline"}
          tone={caps.ollama_available ? "good" : "warn"}
        />
        <Chip
          label="Planner LLM"
          value={caps.planner_llm_configured ? (caps.planner_llm_model ?? "configured") : "none"}
          tone={caps.planner_llm_configured ? "good" : "warn"}
        />
        <Chip
          label="HF push"
          value={caps.hf_push_enabled ? "enabled" : "disabled"}
          tone={caps.hf_push_enabled ? "good" : "neutral"}
        />
      </div>

      {caps.warnings.length > 0 && (
        <ul className="caps__warnings">
          {caps.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Chip({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <span className={`chip chip--${tone}`}>
      <span className="chip__label">{label}</span>
      <span className="chip__value">{value}</span>
    </span>
  );
}
