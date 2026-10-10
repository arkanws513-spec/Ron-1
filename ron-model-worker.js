import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL_ID = "onnx-community/Qwen3-0.6B-ONNX";
let generator = null;
let loading = false;

env.allowRemoteModels = true;
env.useBrowserCache = true;

self.onmessage = async (event) => {
  const { type, messages } = event.data || {};

  if (type === "load") {
    if (generator || loading) return;
    loading = true;
    try {
      const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
      const device = hasWebGPU ? "webgpu" : "wasm";
      const dtype = hasWebGPU ? "q4f16" : "q4";
      self.postMessage({ type: "status", text: hasWebGPU
        ? "جاري تحميل نموذج Qwen3-0.6B على معالج الرسوميات…"
        : "لا يتوفر WebGPU؛ سأحاول تشغيل النموذج محليًا عبر المعالج، وقد يكون بطيئًا…" });

      generator = await pipeline("text-generation", MODEL_ID, {
        device,
        dtype,
        progress_callback: (info) => {
          if (info && info.status) self.postMessage({ type: "progress", info });
        },
      });
      self.postMessage({ type: "ready", text: "النموذج جاهز داخل المتصفح" });
    } catch (error) {
      generator = null;
      self.postMessage({ type: "error", text: "تعذر تحميل النموذج: " + (error?.message || String(error)) });
    } finally {
      loading = false;
    }
    return;
  }

  if (type === "generate") {
    if (!generator) {
      self.postMessage({ type: "error", text: "حمّل النموذج أولًا." });
      return;
    }
    try {
      const result = await generator(messages, {
        max_new_tokens: 160,
        do_sample: true,
        temperature: 0.7,
        repetition_penalty: 1.08,
      });
      let generated = result?.[0]?.generated_text;
      let answer = "";
      if (Array.isArray(generated)) {
        const lastAssistant = [...generated].reverse().find((item) => item?.role === "assistant");
        answer = typeof lastAssistant?.content === "string" ? lastAssistant.content : "";
      } else {
        answer = String(generated || "");
      }
      answer = answer.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\|im_end\|>/g, "").trim();
      self.postMessage({ type: "answer", text: answer || "لم ينتج النموذج إجابة واضحة؛ جرّب صياغة السؤال مرة أخرى." });
    } catch (error) {
      self.postMessage({ type: "error", text: "تعذر توليد الرد: " + (error?.message || String(error)) });
    }
  }
};
