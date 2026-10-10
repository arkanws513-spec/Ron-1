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
  assert.ok(statSync(modelDir + file).size > 0, `missing model metadata: ${file}`);
}

const chunks = [
  ["weights/ron1-q4f16-00.bin", 40000000],
  ["weights/ron1-q4f16-01.bin", 40000000],
  ["weights/ron1-q4f16-02.bin", 37266133],
];
for (const [path, expected] of chunks) {
  assert.equal(statSync(path).size, expected, `unexpected model chunk size: ${path}`);
}
assert.equal(chunks.reduce((sum, [, size]) => sum + size, 0), 117266133);
const q4f16Config = JSON.parse(readFileSync(modelDirV2 + "config.json", "utf8"));
assert.equal(q4f16Config["transformers.js_config"].kv_cache_dtype.q4f16, "float32");

assert.match(worker, /AutoTokenizer\.from_pretrained\(MODEL_ID/);
assert.match(worker, /AutoModelForCausalLM\.from_pretrained\(MODEL_ID/);
assert.match(worker, /type === "generate"/);
assert.match(worker, /max_new_tokens: 80/);
assert.match(worker, /type: "ready"/);
assert.match(worker, /type: "answer"/);
assert.match(html, /ron-model-worker\.js/);
assert.match(html, /function maybeAutoLoad\(\)\{startWorker\(\);/);
assert.match(html, /يجري تشغيل نواة رون تلقائيًا/);
assert.match(html, /<button id="load" type="button" hidden>إعادة تشغيل النواة<\/button>/);
assert.match(pages, /EXPECTED_SHA256="662d0a9d8d5d56e3746a5bf3b3ede96bd2d4d3594d9b2e282baebd4f34cf3589"/);
const serviceWorker = readFileSync("sw.js", "utf8");
assert.match(worker, /use_cache: false/);
assert.match(worker, /generation_reset/);
assert.match(worker, /Unexpected input data type/);
assert.match(html, /navigator\\.serviceWorker\\.register/);
assert.match(html, /generation_reset/);
assert.match(serviceWorker, /caches\\.open/);
assert.match(serviceWorker, /cache\\.put/);
assert.match(pages, /cp index\\.html ron-model-worker\\.js sw\\.js site\\//);
console.log("Ron-1 runtime contract and model asset checks passed.");
