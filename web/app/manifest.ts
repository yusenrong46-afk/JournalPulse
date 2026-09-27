import type { MetadataRoute } from "next";

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "JournalPulse",
    short_name: "JournalPulse",
    description: "Check in with Luna, a calm companion for one small step at a time.",
    start_url: "/",
    display: "standalone",
    background_color: "#fbf5ec",
    theme_color: "#fbf5ec",
    icons: [
      { src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
      { src: "/icon-maskable.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
    ],
  };
}
