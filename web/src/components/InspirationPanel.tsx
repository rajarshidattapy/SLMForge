import { ArrowRightIcon, BookIcon } from "../icons";
import { TEMPLATES, type Template } from "../templates";

type Props = {
  onBrowse: () => void;
  onPick: (template: Template) => void;
};

/** Footer of the home view: points at the templates instead of leaving a dead end. */
export function InspirationPanel({ onBrowse, onPick }: Props) {
  return (
    <section className="panel panel--inspiration">
      <span className="inspiration__glyph" aria-hidden="true">
        <BookIcon size={22} />
      </span>
      <div className="inspiration__copy">
        <h2 className="inspiration__title">Need inspiration?</h2>
        <p className="muted">Check out example builds and templates to get started faster.</p>
      </div>

      <button type="button" className="btn btn--ghost" onClick={onBrowse}>
        Browse templates <ArrowRightIcon size={15} />
      </button>

      <ul className="inspiration__cards">
        {TEMPLATES.slice(0, 3).map((template) => {
          const Icon = template.icon;
          return (
            <li key={template.id}>
              <button type="button" className="mini-card" onClick={() => onPick(template)}>
                <span className="mini-card__icon" aria-hidden="true">
                  <Icon size={17} />
                </span>
                <span className="mini-card__label">{template.short}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
