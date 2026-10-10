const CACHE_NAME = "ron1-native-ui-v1";
self.addEventListener("install", event => event.waitUntil(self.skipWaiting()));
self.addEventListener("activate", event => event.waitUntil((async () => {
  const keys = await caches.keys();
  await Promise.all(keys.filter(key => key.startsWith("ron1-") && key !== CACHE_NAME).map(key => caches.delete(key)));
  await self.clients.claim();
})()));
self.addEventListener("fetch", event => {
  const request = event.request;
  if (request.method !== "GET" || new URL(request.url).origin !== self.location.origin) return;
  const url = new URL(request.url);
  if (url.pathname.endsWith("/sw.js")) return;
  const isShell = request.mode === "navigate" || /\/index\.html$/.test(url.pathname);
  if (!isShell) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    try {
      const fresh = await fetch(request);
      if (fresh.ok) event.waitUntil(cache.put(request, fresh.clone()).catch(() => undefined));
      return fresh;
    } catch (error) {
      const cached = await cache.match(request);
      if (cached) return cached;
      throw error;
    }
  })());
});
