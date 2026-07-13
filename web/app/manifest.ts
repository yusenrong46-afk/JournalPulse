import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "JournalPulse",
    short_name: "JournalPulse",
    description: "A personal laboratory for adaptive reflection.",
    start_url: "/",
    display: "standalone",
    background_color: "#f2ede2",
    theme_color: "#f2ede2",
    icons: [
      { src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
      { src: "/icon-maskable.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
    ],
  };
}
