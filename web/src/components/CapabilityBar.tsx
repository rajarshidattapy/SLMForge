import type { Capabilities } from "../api";
import { RocketIcon, SearchIcon } from "../icons";

type Props = { caps: Capabilities | null };

/** Shows what this machine can actually do, and warns before a build fails later. */
export function CapabilityBar({ caps }: Props) {
  if (!caps) return <div className="caps caps--loading">Checking environment…</div>;

  const compute = caps.gpu_available
    ? `${caps.gpu_name ?? "GPU"} · ${caps.gpu_vram_gb ?? "?"} GB`
    : "CPU only";
  const torch = caps.torch_available ? (caps.torch_device ?? "ready") : "not installed";

  const rows = [
    { label: "Compute", value: compute, warn: !caps.gpu_available },
    { label: "Torch", value: torch, warn: !caps.torch_available },
    { label: "Sandbox", value: caps.sandbox_backend, warn: caps.sandbox_backend !== "docker" },
    {
      label: "Ollama",
      value: caps.ollama_available ? `${caps.ollama_models.length} models` : "offline",
      warn: !caps.ollama_available,
    },
    {
      label: "Planner LLM",
      value: caps.planner_llm_configured ? (caps.planner_llm_model ?? "configured") : "none",
      warn: !caps.planner_llm_configured,
    },
    {
      label: "HF push",
      value: caps.hf_push_enabled ? "enabled" : "disabled",
      warn: false,
    },
  ];

  return (
    <section className="caps">
      <dl className="caps__rows">
        {rows.map((row) => (
          <div key={row.label} className="caps__row">
            <dt>{row.label}</dt>
            <dd className={row.warn ? "caps__value caps__value--warn" : "caps__value"}>
              {row.value}
            </dd>
          </div>
        ))}
      </dl>

      {caps.warnings.length > 0 && (
        <ul className="caps__warnings">
          {caps.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <p className="caps__foot">
        <SearchIcon size={14} /> Python {caps.python_version} on {caps.platform}
      </p>
      {caps.sandbox_backend !== "docker" && (
        <p className="caps__foot caps__foot--warn">
          <RocketIcon size={14} /> {caps.sandbox_backend} sandbox is process isolation, not a
          security boundary.
        </p>
      )}
    </section>
  );
}
