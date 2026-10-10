import { env, AutoTokenizer, AutoModelForCausalLM } from "./vendor/transformers/transformers.min.js";

// Runtime library, tokenizer/configuration files, WASM and weights are all served by GitHub.
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL("./models/", self.location.href).href;
env.useBrowserCache = true;
env.useWasmCache = true;
env.backends.onnx.wasm.wasmPaths = new URL("./vendor/transformers/", self.location.href).href;
// Keep the WebAssembly runtime conservative for phones and memory-limited browsers.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

const MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX";
const GITHUB_Q4_WEIGHTS_URL = "https://github.com/arkanws513-spec/Ron-1/releases/download/ron1-smollm2-135m-q4-v1/Ron-1-Smollm2-135M-Instruct-Q4.onnx";
const originalFetch = globalThis.fetch.bind(globalThis);

// Enforce a GitHub-only runtime: local GitHub Pages files and this GitHub Release are allowed.
const githubOnlyFetch = (input, init) => {
  const requestUrl = typeof input === "string" || input instanceof URL ? String(input) : input?.url;
  if (!requestUrl) throw new Error("Ron-1 blocked a request without a URL.");
  const url = new URL(requestUrl, self.location.href);
  if (url.pathname.includes("/models/onnx-community/SmolLM2-135M-Instruct-ONNX/onnx/model_q4.onnx")) {
    return originalFetch(GITHUB_Q4_WEIGHTS_URL, init);
  }
  if (url.origin === self.location.origin || url.href === GITHUB_Q4_WEIGHTS_URL) {
    return originalFetch(input, init);
  }
  throw new Error("Ron-1 blocked an external runtime request: " + url.origin);
};
globalThis.fetch = githubOnlyFetch;
env.fetch = githubOnlyFetch;

let tokenizer = null;
let model = null;
let loading = false;
let generating = false;

self.onmessage = async (event) => {
  const { type, messages } = event.data || {};
  if (type === "load") {
    if (model || loading) return;
    loading = true;
    try {
      self.postMessage({ type: "status", text: "جاري تشغيل نواة Ron-1 المبنية على SmolLM2-135M بصيغة Q4. سيُعاد استخدام الملفات المخزنة في المتصفح متى أمكن، وقد يلزم تنزيلها إذا لم تكن متاحة في الذاكرة المحلية." });
      const loadedTokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: (info) => { if (info && info.status) self.postMessage({ type: "progress", info }); },
      });
      const loadedModel = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        device: "wasm", dtype: "q4",
        progress_callback: (info) => { if (info && info.status) self.postMessage({ type: "progress", info }); },
      });
      tokenizer = loadedTokenizer;
      model = loadedModel;
      self.postMessage({ type: "ready", text: "النموذج جاهز داخل المتصفح" });
    } catch (error) {
      tokenizer = null; model = null;
      const details = [error?.name, error?.message || String(error), error?.stack].filter(Boolean).join("\\n");
      self.postMessage({ type: "error", text: "تعذر تحميل النموذج على هذا الجهاز. قد يكون السبب حدّ ذاكرة المتصفح أو عدم توافق WebAssembly. التفاصيل: " + details + ". لم نبدأ تنزيلًا ثانيًا تلقائيًا؛ حدّث الصفحة وجرب متصفحًا حديثًا أو جهازًا بذاكرة أكبر، ثم أرسل التفاصيل إن استمر الخطأ." });
    } finally { loading = false; }
    return;
  }
  if (type === "generate") {
    if (!model || !tokenizer || generating) {
      self.postMessage({ type: "error", text: generating ? "رون ما زال يجهز الرد السابق." : "حمّل النموذج أولًا." }); return;
    }
    generating = true;
    try {
      const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });
      const output = await model.generate({ ...inputs, max_new_tokens: 64, do_sample: true, top_k: 20, temperature: 0.7, repetition_penalty: 1.08 });
      const allTokens = output?.tolist?.()[0] || [];
      const inputLength = inputs?.input_ids?.dims?.[1] || 0;
      const answer = tokenizer.decode(allTokens.slice(inputLength), { skip_special_tokens: true }).trim();
      self.postMessage({ type: "answer", text: answer || "لم ينتج النموذج إجابة واضحة؛ جرّب صياغة السؤال مرة أخرى." });
    } catch (error) {
      self.postMessage({ type: "error", text: "تعذر توليد الرد: " + (error?.message || String(error)) });
    } finally { generating = false; }
  }
};
