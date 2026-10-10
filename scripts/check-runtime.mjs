import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const html = readFileSync("index.html", "utf8");
const pages = readFileSync(".github/workflows/pages.yml", "utf8");
const sw = readFileSync("sw.js", "utf8");
const server = readFileSync("core/server.py", "utf8");
const coreReadme = readFileSync("core/README.md", "utf8");

assert.match(html, /Ron-1 Native Core/);
assert.match(html, /fetch\(apiBase\+'\/health'/);
assert.match(html, /fetch\(apiBase\+'\/api\/chat'/);
assert.match(html, /ron1\.nativeApi\.v1/);
assert.doesNotMatch(html, /SmolLM2|Transformers\.js|ron-model-worker\.js|onnx-community/i);
assert.match(html, /لا يوجد أي تحويل تلقائي إلى SmolLM2/);
assert.match(pages, /cp index\.html sw\.js site\//);
assert.doesNotMatch(pages, /huggingface|SmolLM2|onnx\/model_q4|ron1-q4|vendor\/transformers/i);
assert.match(sw, /ron1-native-ui-v1/);
assert.doesNotMatch(sw, /weights|models-v2|vendor\/transformers/i);
assert.match(server, /ron1-native-state-dict-v1/);
assert.match(server, /checkpoint\.get\("trained"\) is not True/);
assert.match(server, /\/api\/chat/);
assert.match(server, /\/health/);
assert.match(coreReadme, /random starting weights/);
assert.match(coreReadme, /not been migrated/);
console.log("Ron-1 native-core interface separation checks passed.");
