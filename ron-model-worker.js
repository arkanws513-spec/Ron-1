import { env, AutoTokenizer, AutoModelForCausalLM, TextStreamer } from "./vendor/transformers/transformers.min.js";

// Runtime library, tokenizer/configuration files, WASM and weights are served from Ron's GitHub repository/release.
env.allowLocalModels = true;
env.allowRemoteModels = true; // Enable fallback for the missing weights file; ronFetch restricts all external fetches below.
env.localModelPath = new URL("./models-v2/", self.location.href).href;
env.useBrowserCache = true;
env.useWasmCache = true;
env.backends.onnx.wasm.wasmPaths = new URL("./vendor/transformers/", self.location.href).href;
// Prefer the lowest-resource WASM execution path for older Android devices.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
// Ask ONNX Runtime for diagnostic messages while opening the model session.
try { env.backends.onnx.logLevel = "verbose"; } catch {}

const MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX";
async function fetchQ4FromPages(init) {
  // The manifest and every binary shard are deployed with Ron-1 on the same origin.
  const manifestUrl = new URL("./weights/ron1-q4-manifest.json", self.location.href);
  const manifestResponse = await originalFetch(manifestUrl, { signal: init?.signal });
  if (!manifestResponse.ok) throw new Error("Failed to load Ron-1 Q4 manifest: HTTP " + manifestResponse.status);
  const manifest = await manifestResponse.json();
  if (!Array.isArray(manifest.chunks) || !Number.isSafeInteger(manifest.totalBytes) || manifest.totalBytes < 1000000) {
    throw new Error("Invalid Ron-1 Q4 manifest");
  }
  const chunks = manifest.chunks.map((item) => {
    if (!item || !/^ron1-q4-\d{2}\.bin$/.test(item.file) || !Number.isSafeInteger(item.size) || item.size <= 0) {
      throw new Error("Invalid entry in Ron-1 Q4 manifest");
    }
    return { url: new URL("./weights/" + item.file, self.location.href).href, size: item.size };
  });
  if (chunks.reduce((sum, item) => sum + item.size, 0) !== manifest.totalBytes) {
    throw new Error("Ron-1 Q4 manifest size mismatch");
  }

  let chunkIndex = 0;
  let currentChunk = null;
  let currentChunkBytes = 0;
  let reader = null;
  let totalBytes = 0;
  const body = new ReadableStream({
    async pull(controller) {
      try {
        while (true) {
          if (!reader) {
            if (chunkIndex >= chunks.length) {
              if (totalBytes !== manifest.totalBytes) {
                controller.error(new Error("Incomplete Ron-1 Q4 model: expected " + manifest.totalBytes + " bytes, received " + totalBytes));
              } else {
                controller.close();
              }
              return;
            }
            currentChunk = chunks[chunkIndex++];
            currentChunkBytes = 0;
            const response = await originalFetch(currentChunk.url, { signal: init?.signal });
            if (!response.ok) throw new Error("Failed to fetch a Ron-1 Q4 chunk: HTTP " + response.status);
            if (!response.body) throw new Error("Ron-1 Q4 chunk has no readable response body");
            reader = response.body.getReader();
          }
          const part = await reader.read();
          if (part.done) {
            if (currentChunkBytes !== currentChunk.size) {
              throw new Error("Incomplete Ron-1 Q4 chunk: expected " + currentChunk.size + " bytes, received " + currentChunkBytes);
            }
            reader = null;
            currentChunk = null;
            continue;
          }
          currentChunkBytes += part.value.byteLength;
          totalBytes += part.value.byteLength;
          controller.enqueue(part.value);
          return;
        }
      } catch (error) {
        controller.error(error);
      }
    },
    async cancel(reason) {
      try { await reader?.cancel(reason); } catch {}
    }
  });
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "application/octet-stream",
      "Accept-Ranges": "bytes"
    }
  });
}
const originalFetch = globalThis.fetch.bind(globalThis);
// Transformers.js captures env.fetch at import time. Override both fetch entry points.
// The canonical Q4 asset is built into Ron-1's published site bundle. Browser-safe chunks are
// served same-origin from Pages to avoid release CORS failures.
const originalEnvFetch = typeof env.fetch === "function" ? env.fetch.bind(env) : originalFetch;
function ronFetch(input, init, fallback) {
  const requestUrl = typeof input === "string" || input instanceof URL ? String(input) : input?.url;
  if (requestUrl && requestUrl.includes("SmolLM2-135M-Instruct-ONNX") && requestUrl.includes("onnx/model_q4.onnx")) {
    return fetchQ4FromPages(init);
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
      self.postMessage({ type: "status", text: "جاري تشغيل نواة Ron-1 المبنية على SmolLM2-135M بصيغة Q4. سيُعاد استخدام الملفات المخزنة في المتصفح متى أمكن، وقد يلزم تنزيلها إذا لم تكن متاحة محليًا." });
      const loadedTokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: (info) => { if (info && info.status) self.postMessage({ type: "progress", info }); },
      });
      loadingStage = "ONNX model/session initialization";
      const loadedModel = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        device: "wasm", dtype: "q4",
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
      // Some ONNX/WASM builds expose a float16 KV cache while the graph expects float32.
      // Retry only that known dtype mismatch without KV caching; this is slower but avoids a hard failure.
      let output;
      try {
        output = await model.generate({ ...inputs, max_new_tokens: 80, do_sample: false, repetition_penalty: 1.08, streamer });
      } catch (firstError) {
        const detail = String(firstError?.message || firstError);
        if (!/Unexpected input data type|tensor\(float16\).*tensor\(float\)|expected tensor\(float\)/i.test(detail)) throw firstError;
        self.postMessage({ type: "generation_reset" });
        self.postMessage({ type: "status", text: "رصد رون تعارضًا في نوع بيانات ذاكرة الاستدلال؛ يجرب مسار توافق أبطأ…" });
        output = await model.generate({ ...inputs, use_cache: false, max_new_tokens: 48, do_sample: false, repetition_penalty: 1.08, streamer });
      }
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
