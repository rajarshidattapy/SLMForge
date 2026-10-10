import { useEffect, useRef, useState } from "react";
import type { ComponentType } from "react";
import type { Project } from "../api";
import {
  ArrowRightIcon,
  BoltIcon,
  DotsIcon,
  FilePlanIcon,
  ImageIcon,
  LanguagesIcon,
  SearchIcon,
  TableIcon,
  TrashIcon,
} from "../icons";
import type { IconProps } from "../icons";
import { buildSubtitle, statusMeta, taskTypeOf, timeAgo } from "../projectMeta";

/** Tile glyph per task type, so a build's row reads at a glance. */
function buildIcon(taskType: string | null): ComponentType<IconProps> {
  const value = taskType?.toLowerCase() ?? "";
  if (value.includes("classif")) return TableIcon;
  if (value.includes("question") || value.includes("qa")) return SearchIcon;
  if (value.includes("translat")) return LanguagesIcon;
  if (value.includes("extract")) return BoltIcon;
  if (value.includes("image") || value.includes("vision")) return ImageIcon;
  return FilePlanIcon;
}

type Props = {
  projects: Project[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  title: string;
  limit?: number;
  onViewAll?: () => void;
};

/** The build records: what was asked for, where it got to, and how to remove it. */
export function BuildList({
  projects,
  selectedId,
  onSelect,
  onDelete,
  title,
  limit,
  onViewAll,
}: Props) {
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!menuFor) return;
    function close(event: MouseEvent) {
      if (!listRef.current?.contains(event.target as Node)) setMenuFor(null);
    }
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuFor]);

  const visible = limit ? projects.slice(0, limit) : projects;

  return (
    <section className="panel panel--list">
      <header className="panel__head">
        <h2 className="panel__title">{title}</h2>
        {onViewAll && projects.length > visible.length && (
          <button type="button" className="link-btn" onClick={onViewAll}>
            View all <ArrowRightIcon size={15} />
          </button>
        )}
      </header>

      {projects.length === 0 ? (
        <p className="muted">
          No builds yet. Describe the model you need above and generate a plan.
        </p>
      ) : (
        <ul className={`builds ${limit ? "builds--compact" : ""}`} ref={listRef}>
          {visible.map((project) => {
            const Icon = buildIcon(taskTypeOf(project));
            const status = statusMeta(project.status);
            const active = project.id === selectedId;
            return (
              <li key={project.id} className={`build ${active ? "build--active" : ""}`}>
                <button type="button" className="build__open" onClick={() => onSelect(project.id)}>
                  <span className="build__icon" aria-hidden="true">
                    <Icon size={19} />
                  </span>
                  <span className="build__body">
                    <span className="build__name">{project.name}</span>
                    <span className="build__sub">{buildSubtitle(project)}</span>
                  </span>
                </button>

                <span className={`pill pill--${status.tone}`}>{status.label}</span>
                <span className="build__time">{timeAgo(project.created_at)}</span>

                <div className="build__menu">
                  <button
                    type="button"
                    className="icon-btn icon-btn--ghost"
                    aria-label={`Actions for ${project.name}`}
                    aria-expanded={menuFor === project.id}
                    onClick={() => setMenuFor(menuFor === project.id ? null : project.id)}
                  >
                    <DotsIcon size={18} />
                  </button>
                  {menuFor === project.id && (
                    <div className="menu" role="menu">
                      <button
                        type="button"
                        role="menuitem"
                        className="menu__item menu__item--danger"
                        onClick={() => {
                          setMenuFor(null);
                          onDelete(project.id);
                        }}
                      >
                        <TrashIcon size={16} /> Delete build
                      </button>
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
