export type IconName = "home" | "journey" | "me" | "back" | "send" | "more" | "chat" | "close";

const PATHS: Record<IconName, React.ReactNode> = {
  home: (
    <>
      <path d="M4 11.2 12 4.5l8 6.7V19a1.5 1.5 0 0 1-1.5 1.5H15v-5.2H9v5.2H5.5A1.5 1.5 0 0 1 4 19v-7.8Z" />
    </>
  ),
  journey: (
    <>
      <path d="M12 20.5V11" />
      <path d="M12 12.5c0-4 2.6-6.8 6.5-7-.1 4-2.7 6.9-6.5 7Z" />
      <path d="M12 15.5c0-3-2.1-5.2-5.4-5.4.1 3.1 2.2 5.3 5.4 5.4Z" />
    </>
  ),
  me: (
    <>
      <circle cx="12" cy="8.5" r="3.6" />
      <path d="M5 20c.8-3.6 3.6-5.6 7-5.6s6.2 2 7 5.6" />
    </>
  ),
  back: <path d="M14.5 5.5 8 12l6.5 6.5" />,
  send: <path d="M5 12h12M12.5 6.5 18 12l-5.5 5.5" />,
  more: (
    <>
      <circle cx="6" cy="12" r="1.3" />
      <circle cx="12" cy="12" r="1.3" />
      <circle cx="18" cy="12" r="1.3" />
    </>
  ),
  chat: <path d="M5 6.5h14v9.2h-7.5L7.5 19v-3.3H5V6.5Z" />,
  close: <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />,
};

export function Icon({ name }: { name: IconName }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {PATHS[name]}
    </svg>
  );
}
