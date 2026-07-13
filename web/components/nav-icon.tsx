type NavIconName = "today" | "reflect" | "history" | "patterns" | "privacy";

export function NavIcon({ name }: { name: NavIconName }) {
  if (name === "today") {
    return (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="7.5" />
        <path d="M12 7.5v4.8l3.1 1.8" />
      </svg>
    );
  }

  if (name === "reflect") {
    return (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path d="M5.5 18.5h3.2L18 9.2l-3.2-3.2-9.3 9.3v3.2Z" />
        <path d="m13.5 7.3 3.2 3.2M5.5 21h13" />
      </svg>
    );
  }

  if (name === "history") {
    return (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path d="M4.8 8.2A8 8 0 1 1 4 14" />
        <path d="M4.8 4.5v3.7H8.5M12 8v4.5l3 1.7" />
      </svg>
    );
  }

  if (name === "patterns") {
    return (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path d="M4 18.5V6M4 18.5h16" />
        <path d="m7 15 3.2-4 3.1 2.2L18 7.5" />
        <circle cx="7" cy="15" r=".8" />
        <circle cx="10.2" cy="11" r=".8" />
        <circle cx="13.3" cy="13.2" r=".8" />
        <circle cx="18" cy="7.5" r=".8" />
      </svg>
    );
  }

  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M12 3.5 19 6v5.2c0 4.4-2.9 7.7-7 9.3-4.1-1.6-7-4.9-7-9.3V6l7-2.5Z" />
      <path d="M9.3 12.1 11 13.8l3.8-4" />
    </svg>
  );
}
