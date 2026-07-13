import type { NextConfig } from "next";

const staticExport = process.env.JOURNALPULSE_STATIC_EXPORT === "true";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  experimental: { inlineCss: true },
  output: staticExport ? "export" : undefined,
  trailingSlash: staticExport,
  ...(staticExport
    ? {}
    : {
        async headers() {
          return [
            {
              source: "/(.*)",
              headers: [
                { key: "X-Content-Type-Options", value: "nosniff" },
                { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
                { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
                { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
                { key: "X-Frame-Options", value: "DENY" },
              ],
            },
          ];
        },
      }),
};

export default nextConfig;
