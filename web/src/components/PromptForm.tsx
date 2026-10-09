import { useState } from "react";

const EXAMPLE =
  "Create a lightweight model that reads customer support tickets and assigns each to " +
  "Billing, Technical, Account Access, or General. It should return valid JSON and " +
  "prioritize classification accuracy.";

type Props = {
  onSubmit: (prompt: string) => Promise<void>;
  busy: boolean;
  error: string | null;
};

/** The single entry point of the product: describe the model you want. */
export function PromptForm({ onSubmit, busy, error }: Props) {
  const [prompt, setPrompt] = useState("");

  const tooShort = prompt.trim().length < 10;

  return (
    <form
      className="prompt"
      onSubmit={async (event) => {
        event.preventDefault();
        if (tooShort || busy) return;
        await onSubmit(prompt.trim());
        setPrompt("");
      }}
    >
      <label className="prompt__label" htmlFor="prompt">
        Describe the model you need
      </label>
      <textarea
        id="prompt"
        className="prompt__input"
        rows={4}
        placeholder={EXAMPLE}
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        disabled={busy}
      />
      <div className="prompt__actions">
        <button className="btn btn--primary" type="submit" disabled={tooShort || busy}>
          {busy ? "Creating…" : "Create build"}
        </button>
        <button
          className="btn btn--ghost"
          type="button"
          onClick={() => setPrompt(EXAMPLE)}
          disabled={busy}
        >
          Use the ticket example
        </button>
        {tooShort && prompt.length > 0 && (
          <span className="prompt__hint">Add a bit more detail (at least 10 characters).</span>
        )}
      </div>
      {error && <p className="alert alert--error">{error}</p>}
    </form>
  );
}
