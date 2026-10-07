const CACHE = "journalpulse-shell-v6";
const OFFLINE_FALLBACK = "/offline.html";

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.add(OFFLINE_FALLBACK)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key)))),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.pathname.startsWith("/v1/") || url.origin !== self.location.origin) return;

  // The exported Turbopack runtime can reuse chunk names across builds. Never
  // serve an older loader/module graph alongside a new page. Only the standalone
  // offline fallback is cached; private pages and API responses remain uncached.
  if (url.pathname.startsWith("/_next/")) return;

  if (event.request.mode === "navigate") {
    event.respondWith(fetch(event.request).catch(() => caches.match(OFFLINE_FALLBACK)));
  }
});
