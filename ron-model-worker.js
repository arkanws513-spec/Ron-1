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
      const attempts = hasWebGPU
        ? [{ device: "webgpu", dtype: "q4f16", label: "معالج الرسوميات" }, { device: "wasm", dtype: "q4", label: "معالج الجهاز" }]
        : [{ device: "wasm", dtype: "q4", label: "معالج الجهاز" }];
      let lastError = null;

      for (let i = 0; i < attempts.length; i++) {
        const option = attempts[i];
        self.postMessage({ type: "status", text: "جاري تحميل Qwen3-0.6B عبر " + option.label + "… قد يستغرق التحميل الأول بعض الوقت." });
        try {
          generator = await pipeline("text-generation", MODEL_ID, {
            device: option.device,
            dtype: option.dtype,
            progress_callback: (info) => {
              if (info && info.status) self.postMessage({ type: "progress", info });
            },
          });
          break;
        } catch (error) {
          lastError = error;
          generator = null;
          if (i < attempts.length - 1) {
            self.postMessage({ type: "status", text: "تعذر تشغيل نسخة الرسوميات؛ أجرب الآن وضع المعالج…" });
          }
        }
      }

      if (!generator) throw lastError || new Error("تعذر تهيئة النموذج");
      self.postMessage({ type: "ready", text: "النموذج جاهز داخل المتصفح" });
    } catch (error) {
      generator = null;
      self.postMessage({ type: "error", text: "تعذر تحميل النموذج على هذا الجهاز: " + (error?.message || String(error)) });
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
      const generated = result?.[0]?.generated_text;
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
