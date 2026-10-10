import { ArrowRightIcon } from "../icons";
import { TEMPLATES, type Template } from "../templates";

type Props = { onUse: (template: Template) => void };

/**
 * Template gallery.
 *
 * A template only fills the prompt box — nothing is created until the user
 * reviews and submits it.
 */
export function TemplatesView({ onUse }: Props) {
  return (
    <section className="page">
      <header className="page__head">
        <h1 className="page__title">Templates</h1>
        <p className="page__sub">
          Starting points written as prompts. Pick one, edit it, then generate the build plan.
        </p>
      </header>

      <div className="card-grid">
        {TEMPLATES.map((template) => {
          const Icon = template.icon;
          return (
            <article key={template.id} className="tcard">
              <span className="tcard__icon" aria-hidden="true">
                <Icon size={20} />
              </span>
              <h2 className="tcard__name">{template.name}</h2>
              <p className="tcard__desc">{template.description}</p>
              <ul className="tcard__tags">
                {template.tags.map((tag) => (
                  <li key={tag} className="tag">
                    {tag}
                  </li>
                ))}
              </ul>
              <button type="button" className="btn btn--ghost" onClick={() => onUse(template)}>
                Use template <ArrowRightIcon size={15} />
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}
