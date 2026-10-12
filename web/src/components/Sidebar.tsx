import { useState } from "react";
import type { ComponentType } from "react";
import { authLoginUrl, type AuthStatus, type AuthUser, type Capabilities } from "../api";
import type { IconProps } from "../icons";
import { BookIcon, BuildsIcon, ChevronRightIcon, GlobeIcon, GridIcon, HomeIcon } from "../icons";
import { CapabilityBar } from "./CapabilityBar";

export type View = "home" | "builds" | "templates" | "docs";

const NAV: { id: View; label: string; icon: ComponentType<IconProps> }[] = [
  { id: "home", label: "Home", icon: HomeIcon },
  { id: "builds", label: "Builds", icon: BuildsIcon },
  { id: "templates", label: "Templates", icon: GridIcon },
  { id: "docs", label: "Docs", icon: BookIcon },
];

type Props = {
  view: View;
  onNavigate: (view: View) => void;
  caps: Capabilities | null;
  auth: AuthStatus | null;
  onSignOut: () => void;
  buildCount: number;
};

/** Left rail: brand, section navigation, machine readiness and the account chip. */
export function Sidebar({ view, onNavigate, caps, auth, onSignOut, buildCount }: Props) {
  const [showEnv, setShowEnv] = useState(false);

  // The readiness light reflects the same capabilities the build will use, so a
  // red/amber light here is the earliest honest warning the user gets.
  const issues = (caps?.warnings.length ?? 0) > 0 || (caps !== null && !caps.planner_llm_configured);
  const envLabel = caps === null ? "Checking environment…" : issues ? "Environment limited" : "Environment ready";
  const envTone = caps === null ? "wait" : issues ? "warn" : "ready";

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="22" height="22">
            <path d="M12 2.5 20 12l-8 9.5L4 12l8-9.5Z" fill="url(#brand-grad)" />
            <defs>
              <linearGradient id="brand-grad" x1="4" y1="2.5" x2="20" y2="21.5">
                <stop offset="0" stopColor="#4f7cf7" />
                <stop offset="1" stopColor="#7b61ff" />
              </linearGradient>
            </defs>
          </svg>
        </span>
        <span className="sidebar__brand-name">SLMForge</span>
      </div>

      <nav className="sidebar__nav" aria-label="Sections">
        {NAV.map((item) => {
          const Icon = item.icon;
          const active = item.id === view;
          return (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${active ? "nav-item--active" : ""}`}
              aria-current={active ? "page" : undefined}
              onClick={() => onNavigate(item.id)}
            >
              <Icon size={19} />
              <span>{item.label}</span>
              {item.id === "builds" && buildCount > 0 && (
                <span className="nav-item__count">{buildCount}</span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="sidebar__foot">
        <button
          type="button"
          className={`env-chip env-chip--${envTone}`}
          aria-expanded={showEnv}
          onClick={() => setShowEnv((open) => !open)}
        >
          <span className="env-chip__dot" aria-hidden="true" />
          <span className="env-chip__label">{envLabel}</span>
          <ChevronRightIcon size={16} className="env-chip__chevron" />
        </button>

        {showEnv && (
          <div className="sidebar__popover">
            <CapabilityBar caps={caps} />
          </div>
        )}

        <AccountChip auth={auth} onSignOut={onSignOut} />
      </div>
    </aside>
  );
}

/**
 * The bottom chip: the signed-in Hub account, a sign-in button, or a plain
 * notice when the deployment has no OAuth app configured.
 */
function AccountChip({ auth, onSignOut }: { auth: AuthStatus | null; onSignOut: () => void }) {
  if (auth?.authenticated && auth.user) {
    return (
      <div className="account">
        <UserAvatar user={auth.user} />
        <span className="account__text">
          <span className="account__name">{auth.user.name}</span>
          <span className="account__role">@{auth.user.username}</span>
        </span>
        <button type="button" className="account__signout" onClick={onSignOut}>
          Sign out
        </button>
      </div>
    );
  }

  // `enabled` is false until HF_CLIENT_ID/HF_CLIENT_SECRET are set, so say that
  // plainly instead of offering a button that would 503.
  if (auth?.enabled) {
    return (
      <a className="signin-btn" href={authLoginUrl(window.location.origin)}>
        <span className="signin-btn__logo" aria-hidden="true">
          🤗
        </span>
        <span className="signin-btn__label">Sign in with Hugging Face</span>
      </a>
    );
  }

  return (
    <div className="account account--guest">
      <span className="account__avatar" aria-hidden="true">
        <GlobeIcon size={16} />
      </span>
      <span className="account__text">
        <span className="account__name">Local workspace</span>
        <span className="account__role">Sign-in not configured</span>
      </span>
    </div>
  );
}

function UserAvatar({ user }: { user: AuthUser }) {
  if (user.avatar_url) {
    return <img className="account__avatar account__avatar--img" src={user.avatar_url} alt="" />;
  }
  const initial = (user.name || user.username || "?").trim().charAt(0).toUpperCase();
  return (
    <span className="account__avatar" aria-hidden="true">
      {initial}
    </span>
  );
}
