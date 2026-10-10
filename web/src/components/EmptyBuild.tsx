import { ArrowRightIcon, CubeIcon } from "../icons";
import { PHASES } from "../templates";

/** Shown when nothing is selected: what a build will go through, in four steps. */
export function EmptyBuild() {
  return (
    <section className="panel panel--empty">
      <span className="empty__glyph" aria-hidden="true">
        <CubeIcon size={26} />
      </span>
      <h2 className="empty__title">No build selected</h2>
      <p className="empty__text">
        Create a build, or pick one from the list, to see its specification, pipeline progress, logs
        and artifacts.
      </p>

      <ol className="phases">
        {PHASES.map((phase, index) => {
          const Icon = phase.icon;
          return (
            <li key={phase.name} className="phases__item">
              {index > 0 && <ArrowRightIcon size={16} className="phases__arrow" />}
              <span className="phase">
                <span className="phase__icon" aria-hidden="true">
                  <Icon size={19} />
                </span>
                <span className="phase__name">{phase.name}</span>
                <span className="phase__detail">{phase.detail}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
