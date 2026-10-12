/** Starter prompts.

 * Templates are just well-written prompts — the pipeline is prompt-first, so a
 * "template" is a starting point the user can edit, never a hidden preset that
 * changes how the build runs.
 */
import type { ComponentType } from "react";
import type { IconProps } from "./icons";
import {
  BoltIcon,
  ChartIcon,
  CubeIcon,
  FilePlanIcon,
  LanguagesIcon,
  RocketIcon,
  SearchIcon,
  TableIcon,
} from "./icons";

export type Template = {
  id: string;
  name: string;
  /** Two-word label for the compact inspiration cards. */
  short: string;
  description: string;
  tags: string[];
  icon: ComponentType<IconProps>;
  prompt: string;
};

export const TICKET_CLASSIFIER_PROMPT =
  "Create a lightweight model that reads customer support tickets and assigns each to " +
  "Billing, Technical, Account Access, or General. It should return valid JSON and " +
  "prioritize classification accuracy.";

export const TEMPLATES: Template[] = [
  {
    id: "ticket-classifier",
    name: "Support Ticket Classifier",
    short: "Classifier",
    description:
      "Route inbound tickets to the right team. Small, fast, and easy to check against a held-out set.",
    tags: ["classification", "JSON output", "small"],
    icon: TableIcon,
    prompt: TICKET_CLASSIFIER_PROMPT,
  },
  {
    id: "doc-qa",
    name: "Document QA (RAG first)",
    short: "RAG",
    description:
      "Answer questions over your own documents, with citations. Should report whether retrieval alone is enough.",
    tags: ["question answering", "retrieval", "citations"],
    icon: SearchIcon,
    prompt:
      "Build a model that answers questions about internal policy documents and returns the answer " +
      "with the source passage it came from. It must say when the documents do not contain the answer " +
      "instead of guessing. Prefer retrieval over fine-tuning if that reaches the accuracy target.",
  },
  {
    id: "code-slm",
    name: "Code Assistant",
    short: "Code SLM",
    description:
      "Explain and patch functions in one repository's style, returning unified diffs that apply cleanly.",
    tags: ["code", "diff output", "instruction following"],
    icon: FilePlanIcon,
    prompt:
      "Fine-tune a small model on our Python codebase conventions so it can explain a function and " +
      "propose a patch as a unified diff. It should keep public APIs unchanged unless asked, and return " +
      "valid diff syntax every time.",
  },
  {
    id: "invoice-extractor",
    name: "Invoice Field Extractor",
    short: "Extractor",
    description:
      "Turn invoice text into a fixed schema with amounts, dates and vendor names, flagging anything ambiguous.",
    tags: ["extraction", "schema", "structured output"],
    icon: BoltIcon,
    prompt:
      "Create a model that reads invoice text and returns JSON with vendor, invoice_number, issue_date, " +
      "currency and total_amount. If a field is missing or unreadable it must be null and listed in a " +
      "warnings array rather than invented.",
  },
  {
    id: "translation",
    name: "Bengali ↔ English Translation",
    short: "Translate",
    description:
      "Short-message translation that keeps names and numbers intact and reports low-confidence cases.",
    tags: ["translation", "bilingual", "low latency"],
    icon: LanguagesIcon,
    prompt:
      "Build a small bilingual model that translates short customer messages between Bengali and English. " +
      "It should preserve names, order IDs and numbers exactly, and mark segments it is unsure about " +
      "instead of silently guessing.",
  },
  {
    id: "eval-harness",
    name: "Answer Quality Scorer",
    short: "Scorer",
    description:
      "Score draft support replies against a rubric and return a number plus the failing criteria.",
    tags: ["evaluation", "rubric", "scoring"],
    icon: ChartIcon,
    prompt:
      "Train a model that scores a draft support reply from 1 to 5 against our tone and accuracy rubric, " +
      "returning the score, the criteria it failed, and a one-line justification. It must be consistent " +
      "when the same reply is scored twice.",
  },
];

/** The hero's "import from example" starter: build on an existing public dataset. */
export const IMPORT_EXAMPLE_PROMPT =
  "Build a classifier for e-commerce product reviews. Start from a public labelled review dataset, " +
  "predict sentiment as positive, neutral or negative, and report macro F1 against a held-out split " +
  "so I can see whether fine-tuning beats a prompted baseline.";

export const COMPOSER_PLACEHOLDER = TICKET_CLASSIFIER_PROMPT;

/** One-line roadmap strip shown when no build is selected. */
export const PHASES: { name: string; detail: string; icon: ComponentType<IconProps> }[] = [
  { name: "Plan", detail: "Generate plan", icon: FilePlanIcon },
  { name: "Train", detail: "Build model", icon: ChartIcon },
  { name: "Evaluate", detail: "Run evaluation", icon: CubeIcon },
  { name: "Deploy", detail: "Ship endpoint", icon: RocketIcon },
];
