/** Inline stroked icons.

 * One tiny set drawn on a 24×24 grid with `currentColor`, so icons inherit text
 * colour and weight from the component that renders them. Kept local instead of
 * pulling in an icon package: the dashboard uses a dozen glyphs and these are
 * the product's visual vocabulary.
 */
import type { ReactNode, SVGProps } from "react";

export type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 20, children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const HomeIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 10.6 12 3.3l9 7.3" />
    <path d="M5.4 9.6V20a1 1 0 0 0 1 1h11.2a1 1 0 0 0 1-1V9.6" />
    <path d="M9.8 21v-6.2h4.4V21" />
  </Icon>
);

export const BuildsIcon = (p: IconProps) => (
  <Icon {...p}>
    <ellipse cx="12" cy="6.4" rx="8" ry="3.1" />
    <path d="M4 6.4v11.2c0 1.7 3.6 3.1 8 3.1s8-1.4 8-3.1V6.4" />
    <path d="M4 12c0 1.7 3.6 3.1 8 3.1s8-1.4 8-3.1" />
  </Icon>
);

export const GridIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3.5" y="3.5" width="7" height="7" rx="2" />
    <rect x="13.5" y="3.5" width="7" height="7" rx="2" />
    <rect x="3.5" y="13.5" width="7" height="7" rx="2" />
    <rect x="13.5" y="13.5" width="7" height="7" rx="2" />
  </Icon>
);

export const BookIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 7.2C10.4 5.7 8.3 5 4 5v13.4c4.3 0 6.4.7 8 2.2 1.6-1.5 3.7-2.2 8-2.2V5c-4.3 0-6.4.7-8 2.2Z" />
    <path d="M12 7.2v13.4" />
  </Icon>
);

export const SparkleIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3.2l1.9 5.4 5.4 1.9-5.4 1.9L12 17.8l-1.9-5.4L4.7 10.5l5.4-1.9L12 3.2Z" />
    <path d="M18.6 16.2l.7 1.9 1.9.7-1.9.7-.7 1.9-.7-1.9-1.9-.7 1.9-.7.7-1.9Z" />
  </Icon>
);

export const BoltIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M13.4 2.5 5 13.6h5.6L10 21.5l8.6-11.2h-5.8l.6-7.8Z" />
  </Icon>
);

export const TableIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3.5" y="4.5" width="17" height="15" rx="2.5" />
    <path d="M3.5 9.5h17M9.5 9.5v10M3.5 14.5h17" />
  </Icon>
);

export const ImportIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3.5v11" />
    <path d="m7.8 10.3 4.2 4.2 4.2-4.2" />
    <path d="M4.5 19.5h15" />
  </Icon>
);

export const ArrowRightIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4.5 12h14" />
    <path d="m13 6.5 5.5 5.5L13 17.5" />
  </Icon>
);

export const ChevronRightIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m9.5 5.5 6.5 6.5-6.5 6.5" />
  </Icon>
);

export const ChevronDownIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m5.5 9 6.5 6.5L18.5 9" />
  </Icon>
);

export const PaperclipIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M20 11.6 12.2 19.4a5 5 0 0 1-7.1-7.1l7.8-7.8a3.3 3.3 0 0 1 4.7 4.7l-7.8 7.8a1.7 1.7 0 0 1-2.4-2.4l6.8-6.8" />
  </Icon>
);

export const GlobeIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M3.5 12h17" />
    <path d="M12 3.5c2.2 2.4 3.3 5.3 3.3 8.5S14.2 18.1 12 20.5c-2.2-2.4-3.3-5.3-3.3-8.5S9.8 5.9 12 3.5Z" />
  </Icon>
);

export const SlidersIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 7.5h14M5 12h14M5 16.5h14" />
    <circle cx="9" cy="7.5" r="1.9" />
    <circle cx="15" cy="12" r="1.9" />
    <circle cx="10.5" cy="16.5" r="1.9" />
  </Icon>
);

export const ChartIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5.5 20V11" />
    <path d="M12 20V4.5" />
    <path d="M18.5 20v-6" />
  </Icon>
);

export const RocketIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M14.4 4.2c2.9 0 5.4 2.5 5.4 5.4 0 3.8-4.1 7.8-7.4 9.6L4.8 11.6c1.8-3.3 5.8-7.4 9.6-7.4Z" />
    <circle cx="14.8" cy="9.2" r="1.9" />
    <path d="m6.9 17.1-2.4 2.4" />
  </Icon>
);

export const CubeIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m12 3.2 8 4.4v8.8l-8 4.4-8-4.4V7.6l8-4.4Z" />
    <path d="M4 7.6l8 4.4 8-4.4" />
    <path d="M12 12v8.8" />
  </Icon>
);

export const TargetIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 20.5a8.5 8.5 0 1 0-8.5-8.5" />
    <path d="m4.2 12.2 3 3 4.6-5.4" />
    <path d="M16.4 7.2 20 3.6" />
  </Icon>
);

export const FilePlanIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M13.5 3.5H7.5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8.5l-5-5Z" />
    <path d="M13.5 3.5v5h5" />
    <path d="M9 13.5h6M9 17h4" />
  </Icon>
);

export const SearchIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="10.8" cy="10.8" r="6.3" />
    <path d="m15.5 15.5 4.2 4.2" />
  </Icon>
);

export const ImageIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3.5" y="4.5" width="17" height="15" rx="2.5" />
    <circle cx="9" cy="10" r="1.6" />
    <path d="m4.5 17.5 4.4-4 3.2 3 3.4-3.6 4 4.1" />
  </Icon>
);

export const LanguagesIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 6.5h8.5" />
    <path d="M8.2 6.5c0 4.6-1.9 8-4.7 9.8" />
    <path d="M5.7 10.6c1.4 3 3.3 5.2 6 6.6" />
    <path d="m12.8 20.5 3.7-9.5 3.7 9.5" />
    <path d="M14.4 17h4.3" />
  </Icon>
);

export const DotsIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="5.5" r="1.4" fill="currentColor" stroke="none" />
    <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
    <circle cx="12" cy="18.5" r="1.4" fill="currentColor" stroke="none" />
  </Icon>
);

export const TrashIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4.5 7h15" />
    <path d="M9.5 7V4.8h5V7" />
    <path d="M6.6 7l.9 12.2a1.6 1.6 0 0 0 1.6 1.5h5.8a1.6 1.6 0 0 0 1.6-1.5L17.4 7" />
  </Icon>
);

export const CheckIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m5 12.6 4.6 4.6L19 6.8" />
  </Icon>
);

