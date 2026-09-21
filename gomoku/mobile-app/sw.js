// Offline caching strategy:
//   - big binaries (models / wasm / icons): cache-first  (offline + fast)
//   - HTML / CSS / JS:                      network-first (updates apply immediately;
//                                           cache is only the offline fallback)
// On localhost the SW unregisters itself so local dev always gets fresh files.
if (["localhost", "127.0.0.1"].includes(self.location.hostname)) {
  self.registration.unregister();
  self.addEventListener("fetch", () => {});
} else {
  const CACHE = "chess-rl-v10";
  const CACHE_FIRST = [
    "/models/", "/vendor/", "/icons/",
    "gomoku_policy_value.onnx", "xiangqi_policy_value.onnx",
    "ort.min.js", "ort-wasm-simd.wasm", "ort-wasm-simd-threaded.jsep.wasm",
    "icon-192.png", "icon-512.png",
  ];
  const ASSETS = [
    "./", "./index.html",
    "./gomoku.html", "./xiangqi.html", "./go.html",
    "./style.css", "./app.js", "./engine.js", "./ai.js", "./worker.js",
    "./xiangqi-app.js", "./xiangqi-engine.js", "./xiangqi-ai.js", "./xiangqi-worker.js",
    "./vendor/ort.min.js", "./vendor/ort-wasm-simd.wasm",
    "./vendor/ort-wasm-simd-threaded.jsep.wasm",
    "./models/gomoku_policy_value.onnx", "./models/xiangqi_policy_value.onnx",
    "./manifest.webmanifest", "./icons/icon-192.png", "./icons/icon-512.png",
  ];

  self.addEventListener("install", (e) => {
    e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
  });
  self.addEventListener("activate", (e) => {
    e.waitUntil(caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()));
  });
  self.addEventListener("fetch", (e) => {
    const url = e.request.url;
    const cacheFirst = CACHE_FIRST.some((s) => url.includes(s));
    if (cacheFirst) {
      e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
          return resp;
        })));
    } else {
      e.respondWith(fetch(e.request)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
          return resp;
        })
        .catch(() => caches.match(e.request, { ignoreSearch: true })));
    }
  });
}
