import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";

const worker = readFileSync("ron-model-worker.js", "utf8");
const html = readFileSync("index.html", "utf8");
const pages = readFileSync(".github/workflows/pages.yml", "utf8");
const modelDir = "models/onnx-community/SmolLM2-135M-Instruct-ONNX/";
const modelDirV2 = "models-v2/onnx-community/SmolLM2-135M-Instruct-ONNX/";

for (const file of [
  "config.json", "generation_config.json", "merges.txt",
  "quantize_config.json", "special_tokens_map.json", "tokenizer.json",
  "tokenizer_config.json", "vocab.json"
]) {
  assert.ok(statSync(modelDir + file).size > 0, \`missing model metadata: \${file}\`);
}
const modelConfig = JSON.parse(readFileSync(modelDirV2 + "config.json", "utf8"));
assert.equal(modelConfig["transformers.js_config"].kv_cache_dtype.q4, "float32");
assert.equal(modelConfig["transformers.js_config"].kv_cache_dtype.q4f16, undefined);

assert.match(worker, /AutoTokenizer\.from_pretrained\(MODEL_ID/);
assert.match(worker, /AutoModelForCausalLM\.from_pretrained\(MODEL_ID/);
assert.match(worker, /type === "generate"/);
assert.match(worker, /dtype: "q4"/);
assert.match(worker, /onnx\/model_q4\.onnx/);
assert.match(worker, /ron1-q4-manifest\.json/);
assert.match(worker, /Incomplete Ron-1 Q4 chunk/);
assert.match(worker, /max_new_tokens: 80/);
assert.match(worker, /type: "ready"/);
assert.match(worker, /type: "answer"/);
assert.match(html, /ron-model-worker\.js/);
assert.ok(html.includes("async function maybeAutoLoad()"));
assert.match(html, /يجري تشغيل نواة رون تلقائيًا/);
assert.match(html, /<button id="load" type="button" hidden>إعادة تشغيل النواة<\/button>/);
assert.match(pages, /onnx\/model_q4\.onnx/);
assert.match(pages, /ron1-q4-manifest\.json/);
assert.match(pages, /split -b 40000000/);
assert.match(pages, /Q4 model artifact is unexpectedly small/);
const serviceWorker = readFileSync("sw.js", "utf8");
assert.match(worker, /use_cache: false/);
assert.match(worker, /generation_reset/);
assert.match(worker, /Unexpected input data type/);
assert.ok(html.includes("navigator.serviceWorker.register"));
assert.match(html, /generation_reset/);
assert.ok(serviceWorker.includes("caches.open"));
assert.ok(serviceWorker.includes("cache.put"));
assert.match(serviceWorker, /if \(isAppShell\)/);
assert.match(serviceWorker, /const cached = await cache\.match\(request\)/);
assert.match(serviceWorker, /ron1-q4-manifest/);
assert.ok(pages.includes("cp index.html ron-model-worker.js sw.js site/"));
console.log("Ron-1 Q4/WASM core contract and model asset checks passed.");
