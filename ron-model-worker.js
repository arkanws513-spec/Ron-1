import {
  AutoTokenizer,
  AutoModelForCausalLM,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX";
let tokenizer = null;
let model = null;
let loading = false;
let generating = false;

self.onmessage = async (event) => {
  const { type, messages } = event.data || {};

  if (type === "load") {
    if (model || loading) return;
    loading = true;
    const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
    const attempts = hasWebGPU
      ? [{ device: "webgpu", dtype: "q4f16", label: "معالج الرسوميات" }, { device: "wasm", dtype: "q4", label: "معالج الجهاز" }]
      : [{ device: "wasm", dtype: "q4", label: "معالج الجهاز" }];
    let lastError = null;

    for (let i = 0; i < attempts.length; i++) {
      const option = attempts[i];
      try {
        self.postMessage({ type: "status", text: "جاري تحميل Qwen2.5-0.5B عبر " + option.label + "… قد يستغرق التحميل الأول بعض الوقت." });
        const loadedTokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
          progress_callback: (info) => {
            if (info && info.status) self.postMessage({ type: "progress", info });
          },
        });
        const loadedModel = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
          device: option.device,
          dtype: option.dtype,
          progress_callback: (info) => {
            if (info && info.status) self.postMessage({ type: "progress", info });
          },
        });
        tokenizer = loadedTokenizer;
        model = loadedModel;
        break;
      } catch (error) {
        lastError = error;
        tokenizer = null;
        model = null;
        if (i < attempts.length - 1) {
          self.postMessage({ type: "status", text: "تعذر تشغيل نسخة الرسوميات؛ أجرب الآن وضع المعالج…" });
        }
      }
    }

    if (model && tokenizer) {
      self.postMessage({ type: "ready", text: "النموذج جاهز داخل المتصفح" });
    } else {
      self.postMessage({ type: "error", text: "تعذر تحميل النموذج على هذا الجهاز: " + (lastError?.message || "خطأ غير معروف") });
    }
    loading = false;
    return;
  }

  if (type === "generate") {
    if (!model || !tokenizer || generating) {
      self.postMessage({ type: "error", text: generating ? "رون ما زال يجهز الرد السابق." : "حمّل النموذج أولًا." });
      return;
    }
    generating = true;
    try {
      const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
        
      });
      const output = await model.generate({
        ...inputs,
        max_new_tokens: 64,
        do_sample: true,
        top_k: 20,
        temperature: 0.7,
        repetition_penalty: 1.08,
      });
      const allTokens = output?.tolist?.()[0] || [];
      const inputLength = inputs?.input_ids?.dims?.[1] || 0;
      const answerTokens = allTokens.slice(inputLength);
      const answer = tokenizer.decode(answerTokens, { skip_special_tokens: true }).trim();
      self.postMessage({ type: "answer", text: answer || "لم ينتج النموذج إجابة واضحة؛ جرّب صياغة السؤال مرة أخرى." });
    } catch (error) {
      self.postMessage({ type: "error", text: "تعذر توليد الرد: " + (error?.message || String(error)) });
    } finally {
      generating = false;
    }
  }
};
