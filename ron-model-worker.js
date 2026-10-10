import { env, AutoTokenizer, AutoModelForCausalLM, TextStreamer } from "./vendor/transformers/transformers.min.js";

// Runtime library, tokenizer/configuration files, WASM and weights are served from Ron's GitHub repository/release.
env.allowLocalModels = true;
env.allowRemoteModels = true; // Enable fallback for the missing weights file; ronFetch restricts all external fetches below.
env.localModelPath = new URL("./models/", self.location.href).href;
env.useBrowserCache = true;
env.useWasmCache = true;
env.backends.onnx.wasm.wasmPaths = new URL("./vendor/transformers/", self.location.href).href;
// Prefer the lowest-resource WASM execution path for older Android devices.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
// Ask ONNX Runtime for diagnostic messages while opening the model session.
try { env.backends.onnx.logLevel = "verbose"; } catch {}

const MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX";
const GITHUB_Q4F16_ASSET_API_URL = "https://api.github.com/repos/arkanws513-spec/Ron-1/releases/assets/627890887";
const originalFetch = globalThis.fetch.bind(globalThis);
// Transformers.js captures env.fetch at import time. Override both fetch entry points.
// Only the exact Q4F16 weights file may leave GitHub Pages, and it is redirected to Ron-1's
// GitHub Release. Every other request must remain same-origin (or be a local blob/data URL).
const originalEnvFetch = typeof env.fetch === "function" ? env.fetch.bind(env) : originalFetch;
function ronFetch(input, init, fallback) {
  const requestUrl = typeof input === "string" || input instanceof URL ? String(input) : input?.url;
  if (requestUrl && requestUrl.includes("SmolLM2-135M-Instruct-ONNX") && requestUrl.includes("onnx/model_q4f16.onnx")) {
    const headers = new Headers(init?.headers || (input instanceof Request ? input.headers : undefined));
    headers.set("Accept", "application/octet-stream");
    return originalFetch(GITHUB_Q4F16_ASSET_API_URL, { ...init, headers });
  }
  if (requestUrl) {
    if (requestUrl.startsWith("blob:") || requestUrl.startsWith("data:")) return fallback(input, init);
    try {
      const parsed = new URL(requestUrl, self.location.href);
      if (parsed.origin === self.location.origin) return fallback(input, init);
    } catch {}
  }
  throw new Error("Blocked unexpected external asset request: " + String(requestUrl || input));
}
globalThis.fetch = (input, init) => ronFetch(input, init, originalFetch);
env.fetch = (input, init) => ronFetch(input, init, originalEnvFetch);

let tokenizer = null;
let model = null;
let loading = false;
let generating = false;
let loadingStage = "idle";
let runtimeDiagnostics = [];
let originalConsoleMethods = null;

function beginRuntimeDiagnostics() {
  runtimeDiagnostics = [];
  originalConsoleMethods = {};
  for (const level of ["error", "warn", "info", "log"]) {
    const original = console[level]?.bind(console);
    if (!original) continue;
    originalConsoleMethods[level] = console[level];
    console[level] = (...args) => {
      try {
        const line = args.map((value) => typeof value === "string" ? value : String(value?.message || value)).join(" ").slice(0, 600);
        if (line && runtimeDiagnostics.length < 12) runtimeDiagnostics.push(level + ": " + line);
      } catch {}
      original(...args);
    };
  }
}

function endRuntimeDiagnostics() {
  if (!originalConsoleMethods) return;
  for (const [level, method] of Object.entries(originalConsoleMethods)) console[level] = method;
  originalConsoleMethods = null;
}

function describeError(error) {
  if (typeof error === "number") {
    return "numeric runtime error " + error + " (0x" + (error >>> 0).toString(16) + ")";
  }
  if (typeof error === "string") return error;
  return [error?.name, error?.message, error?.stack].filter(Boolean).join("\n") || String(error);
}

self.onmessage = async (event) => {
  const { type, messages } = event.data || {};
  if (type === "load") {
    if (model || loading) return;
    loading = true;
    loadingStage = "tokenizer";
    beginRuntimeDiagnostics();
    try {
      self.postMessage({ type: "status", text: "جاري تشغيل نواة Ron-1 المبنية على SmolLM2-135M بصيغة Q4F16. سيُعاد استخدام الملفات المخزنة في المتصفح متى أمكن، وقد يلزم تنزيلها إذا لم تكن متاحة محليًا." });
      const loadedTokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: (info) => { if (info && info.status) self.postMessage({ type: "progress", info }); },
      });
      loadingStage = "ONNX model/session initialization";
      const loadedModel = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        device: "wasm", dtype: "q4f16",
        progress_callback: (info) => { if (info && info.status) self.postMessage({ type: "progress", info }); },
      });
      loadingStage = "finalizing model";
      tokenizer = loadedTokenizer;
      model = loadedModel;
      self.postMessage({ type: "ready", text: "النموذج جاهز داخل المتصفح" });
    } catch (error) {
      tokenizer = null;
      model = null;
      const details = describeError(error);
      const diagnostics = runtimeDiagnostics.length ? " | ONNX diagnostics: " + runtimeDiagnostics.slice(-8).join(" || ") : " | لم يُصدر ONNX Runtime تفاصيل نصية إضافية.";
      self.postMessage({ type: "error", text: "فشل التحميل في مرحلة " + loadingStage + ": " + details + diagnostics + ". لم يبدأ تنزيلًا ثانيًا تلقائيًا." });
    } finally {
      endRuntimeDiagnostics();
      loading = false;
      loadingStage = "idle";
    }
    return;
  }
  if (type === "generate") {
    if (!model || !tokenizer || generating) {
      self.postMessage({ type: "error", text: generating ? "رون ما زال يجهز الرد السابق." : "حمّل النموذج أولًا." });
      return;
    }
    generating = true;
    try {
      self.postMessage({ type: "generation_started", text: "بدأ رون توليد الرد" });
      const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => {
          if (text) self.postMessage({ type: "token", text });
        },
      });
      // Short, deterministic generations are more responsive on low-memory mobile CPUs.
      const output = await model.generate({ ...inputs, max_new_tokens: 40, do_sample: false, repetition_penalty: 1.08, streamer });
      const allTokens = output?.tolist?.()[0] || [];
      const inputLength = inputs?.input_ids?.dims?.[1] || 0;
      const answer = tokenizer.decode(allTokens.slice(inputLength), { skip_special_tokens: true }).trim();
      self.postMessage({ type: "answer", text: answer || "لم ينتج النموذج إجابة واضحة؛ جرّب صياغة السؤال مرة أخرى." });
    } catch (error) {
      self.postMessage({ type: "error", text: "تعذر توليد الرد: " + (error?.message || String(error)) });
    } finally {
      generating = false;
    }
  }
};
