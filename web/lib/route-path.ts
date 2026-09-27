import { usePathname } from "next/navigation";

// The static export uses trailing slashes, so the same page can be reported as
// "/login" or "/login/". Route checks compare against the unslashed form.
export function normalizeRoutePath(pathname: string | null): string {
  if (!pathname) return "/";
  const trimmed = pathname.replace(/\/+$/, "");
  return trimmed || "/";
}

export function useRoutePath(): string {
  return normalizeRoutePath(usePathname());
}
