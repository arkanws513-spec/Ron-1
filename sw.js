const CACHE_NAME = "ron1-runtime-v2";

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

  const isAppShell = request.mode === "navigate" || /\/(?:index\.html|ron-model-worker\.js)$/.test(url.pathname);
  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    if (isAppShell) {
      // Keep the interface and worker updatable; use their cached copy only when offline.
      try {
        const fresh = await fetch(request);
        if (fresh.ok) event.waitUntil(cache.put(request, fresh.clone()).catch(() => undefined));
        return fresh;
      } catch (error) {
        const cachedShell = await cache.match(request);
        if (cachedShell) return cachedShell;
        throw error;
      }
    }

    // Heavy, versioned runtime/model assets use cache-first to support repeat and offline loads.
    const cached = await cache.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok && response.type !== "opaque" && request.headers.get("range") === null) {
      const isRuntimeAsset = /\/(?:vendor\/transformers\/|models-v2?\/|weights\\/ron1-q4-\\d{2}\\.bin)/.test(url.pathname);
      if (isRuntimeAsset) {
        // Large weight chunks may exceed a device's storage quota. Cache failures must never break inference.
        event.waitUntil(cache.put(request, response.clone()).catch(() => undefined));
      }
    }
    return response;
  })());
});
