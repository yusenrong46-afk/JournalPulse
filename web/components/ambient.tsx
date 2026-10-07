/**
 * Paper & Lamp ambience: a drawn branch shadow and a warm lamp glow at the page edges.
 * Purely decorative, kept behind content and hidden from assistive technology.
 */
export function Ambient() {
  return (
    <div className="ambient" aria-hidden="true">
      <span className="ambient-lamp" />
      <span className="ambient-sage" />
      <svg className="ambient-branch" viewBox="0 0 260 260" focusable="false">
        <path d="M10 250 C90 170 160 110 250 10" />
        <ellipse cx="70" cy="190" rx="26" ry="9" transform="rotate(-55 70 190)" />
        <ellipse cx="100" cy="150" rx="26" ry="9" transform="rotate(-20 100 150)" />
        <ellipse cx="130" cy="130" rx="26" ry="9" transform="rotate(-70 130 130)" />
        <ellipse cx="165" cy="90" rx="26" ry="9" transform="rotate(-25 165 90)" />
        <ellipse cx="190" cy="75" rx="24" ry="8" transform="rotate(-75 190 75)" />
        <ellipse cx="220" cy="40" rx="22" ry="8" transform="rotate(-30 220 40)" />
      </svg>
    </div>
  );
}
