import { AutoTokenizer, AutoModelForCausalLM } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX";
// Keep the app on GitHub Pages and fetch the Q4 ONNX weights from this repo's GitHub Release.
// Tokenizer/config files still come from the upstream model repository.
const GITHUB_Q4_WEIGHTS_URL = "https://github.com/arkanws513-spec/Ron-1/releases/download/ron1-smollm2-135m-q4-v1/Ron-1-Smollm2-135M-Instruct-Q4.onnx";
const originalFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = (input, init) => {
  const requestUrl = typeof input === "string" ? input : input?.url;
  if (requestUrl && requestUrl.includes("huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX/resolve/") && requestUrl.includes("/onnx/model_q4.onnx")) {
    return originalFetch(GITHUB_Q4_WEIGHTS_URL, init);
  }
  return originalFetch(input, init);
};

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
      self.postMessage({ type: "status", text: "جاري تجهيز SmolLM2-135M بصيغة Q4 على معالج الجهاز. قد يستغرق التحميل الأول بعض الوقت." });
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
      self.postMessage({ type: "error", text: "تعذر تحميل النموذج على هذا الجهاز: " + (error?.message || String(error)) + ". لم نبدأ تنزيلًا ثانيًا تلقائيًا؛ أرسل نص الخطأ لفحصه." });
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
