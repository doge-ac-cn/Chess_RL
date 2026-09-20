// Offline cache: app shell + model + runtime. Version bump to refresh.
const CACHE = "gomoku-ai-v6";
const ASSETS = [
  "./", "./index.html", "./style.css", "./app.js",
  "./engine.js", "./ai.js", "./worker.js",
  "./vendor/ort.min.js", "./vendor/ort-wasm-simd.wasm", "./vendor/ort-wasm-simd-threaded.jsep.wasm",
  "./models/gomoku_policy_value.onnx",
  "./manifest.webmanifest", "./icons/icon-192.png", "./icons/icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request)));
});
