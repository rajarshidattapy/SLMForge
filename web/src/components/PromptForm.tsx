import { useState } from "react";
import {
  ArrowRightIcon,
  GlobeIcon,
  PaperclipIcon,
  SlidersIcon,
  SparkleIcon,
} from "../icons";
import { COMPOSER_PLACEHOLDER, IMPORT_EXAMPLE_PROMPT, TICKET_CLASSIFIER_PROMPT } from "../templates";

type Props = {
  prompt: string;
  onPromptChange: (prompt: string) => void;
  onSubmit: (prompt: string, name: string | null) => Promise<void>;
  busy: boolean;
};

const MIN_LENGTH = 10;

/**
 * The single entry point of the product: describe the model you want.
 *
 * Controls that the backend cannot honour yet stay visibly disabled with the
 * reason in their tooltip, rather than pretending to do something.
 */
export function PromptForm({ prompt, onPromptChange, onSubmit, busy }: Props) {
  const [name, setName] = useState("");
  const [advanced, setAdvanced] = useState(false);

  const tooShort = prompt.trim().length < MIN_LENGTH;

  return (
    <form
      className="composer"
      onSubmit={async (event) => {
        event.preventDefault();
        if (tooShort || busy) return;
        await onSubmit(prompt.trim(), name.trim() || null);
      }}
    >
      <textarea
        id="prompt"
        className="composer__input"
        rows={4}
        aria-label="Describe the model you need"
        placeholder={COMPOSER_PLACEHOLDER}
        value={prompt}
        onChange={(event) => onPromptChange(event.target.value)}
        disabled={busy}
      />

      <div className="composer__row">
        <div className="composer__tools">
          <button
            type="button"
            className="icon-btn"
            disabled
            title="Dataset upload arrives with the curate-data stage"
            aria-label="Attach a dataset (not built yet)"
          >
            <PaperclipIcon size={18} />
          </button>
          <button
            type="button"
            className="chip-btn"
            disabled
            title="Live web context is not built yet — the analyzer reads your prompt only"
          >
            <GlobeIcon size={16} />
            Web context
          </button>
          <button
            type="button"
            className={`chip-btn ${advanced ? "chip-btn--on" : ""}`}
            aria-expanded={advanced}
            onClick={() => setAdvanced((open) => !open)}
          >
            <SlidersIcon size={16} />
            Advanced
          </button>
        </div>

        <button className="btn btn--primary btn--generate" type="submit" disabled={tooShort || busy}>
          <SparkleIcon size={17} />
          {busy ? "Generating…" : "Generate build plan"}
          <ArrowRightIcon size={17} />
        </button>
      </div>

      {advanced && (
        <div className="composer__advanced">
          <label className="field">
            <span className="field__label">Build name (optional)</span>
            <input
              className="field__input"
              type="text"
              value={name}
              placeholder="Support ticket classifier"
              onChange={(event) => setName(event.target.value)}
              disabled={busy}
            />
          </label>
          <div className="field">
            <span className="field__label">Start from an example</span>
            <div className="composer__presets">
              <button
                type="button"
                className="chip-btn"
                onClick={() => onPromptChange(TICKET_CLASSIFIER_PROMPT)}
              >
                Ticket classifier
              </button>
              <button
                type="button"
                className="chip-btn"
                onClick={() => onPromptChange(IMPORT_EXAMPLE_PROMPT)}
              >
                Review sentiment
              </button>
            </div>
          </div>
        </div>
      )}

      {tooShort && prompt.length > 0 && (
        <p className="composer__hint">
          Add a bit more detail (at least {MIN_LENGTH} characters) so the analyzer has something to
          work with.
        </p>
      )}
    </form>
  );
}
