import { ChartIcon, CubeIcon, FilePlanIcon, TargetIcon } from "../icons";

/**
 * Decorative hero illustration: the four build stages orbiting the model.
 *
 * Pure SVG + absolutely-positioned labels — no image asset to ship or load, so
 * it stays crisp at any size and the whole thing is ignored by screen readers.
 */
export function HeroArt() {
  return (
    <div className="art" aria-hidden="true">
      <svg className="art__ridge" viewBox="0 0 560 300" preserveAspectRatio="none">
        <defs>
          <linearGradient id="ridge-far" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#ffffff" stopOpacity="0.6" />
            <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="ridge-near" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#e2e6ff" stopOpacity="0.95" />
            <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon
          points="0,186 92,126 172,172 252,108 342,178 424,138 560,196 560,300 0,300"
          fill="url(#ridge-far)"
        />
        <polygon
          points="0,232 122,174 222,226 322,164 432,226 560,188 560,300 0,300"
          fill="url(#ridge-near)"
        />
      </svg>

      <div className="art__glow" />
      <div className="art__ring" />

      <svg className="art__crystal" viewBox="0 0 200 260">
        <defs>
          <linearGradient id="crystal-top-left" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#dbe7ff" />
            <stop offset="1" stopColor="#8fb0ff" />
          </linearGradient>
          <linearGradient id="crystal-top-right" x1="1" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#b9a9ff" />
            <stop offset="1" stopColor="#7c8cff" />
          </linearGradient>
          <linearGradient id="crystal-bottom-left" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#a89bff" />
            <stop offset="1" stopColor="#6a5cf0" />
          </linearGradient>
          <linearGradient id="crystal-bottom-right" x1="1" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#6f9bff" />
            <stop offset="1" stopColor="#3f4fd6" />
          </linearGradient>
        </defs>
        <polygon points="100,6 100,98 8,66" fill="url(#crystal-top-left)" />
        <polygon points="100,6 192,66 100,98" fill="url(#crystal-top-right)" />
        <polygon points="8,66 100,98 100,254" fill="url(#crystal-bottom-left)" />
        <polygon points="192,66 100,98 100,254" fill="url(#crystal-bottom-right)" />
        <polyline
          points="100,6 100,254"
          fill="none"
          stroke="#ffffff"
          strokeWidth="1.4"
          opacity="0.55"
        />
      </svg>

      <span className="art__chip art__chip--train">
        <ChartIcon size={15} /> Train
      </span>
      <span className="art__chip art__chip--evaluate">
        <TargetIcon size={15} /> Evaluate
      </span>
      <span className="art__chip art__chip--plan">
        <FilePlanIcon size={15} /> Plan
      </span>
      <span className="art__chip art__chip--deploy">
        <CubeIcon size={15} /> Deploy
      </span>
    </div>
  );
}
