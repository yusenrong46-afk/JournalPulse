import type { MetadataRoute } from "next";

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "JournalPulse",
    short_name: "JournalPulse",
    description: "A personal laboratory for adaptive reflection.",
    start_url: "/",
    display: "standalone",
    background_color: "#f1ecdf",
    theme_color: "#f1ecdf",
    icons: [
      { src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
      { src: "/icon-maskable.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
    ],
  };
}
