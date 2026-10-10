const CACHE_NAME = "ron1-runtime-v1";
const MODEL_PATH = /\/(?:weights\/ron1-q4f16-\\d+\\.bin|vendor\/transformers\/|models-v2?\/)/;

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((key) => key.startsWith("ron1-runtime-") && key !== CACHE_NAME).map((key) => caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET" || new URL(request.url).origin !== self.location.origin) return;
  const url = new URL(request.url);
  if (url.pathname.endsWith("/sw.js")) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    if (cached) return cached;

    const response = await fetch(request);
    if (response.ok && response.type !== "opaque" && request.headers.get("range") === null) {
      const isAppAsset = request.mode === "navigate"
        || /\/(?:index\\.html|ron-model-worker\\.js|vendor\/transformers\/|models-v2?\/|weights\/ron1-q4f16-\\d+\\.bin)/.test(url.pathname);
      if (isAppAsset) {
        // Large weight chunks may exceed a device's storage quota. Cache failures must never break inference.
        event.waitUntil(cache.put(request, response.clone()).catch(() => undefined));
      }
    }
    return response;
  })());
});
