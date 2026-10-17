# Ron-1 Vercel inference API

This repository contains lightweight Vercel Python functions that proxy chat requests to Hugging Face Inference Providers. The Hugging Face token stays server-side and is never included in the browser bundle.

## Required setup

1. In Hugging Face token settings, create a fine-grained token with **Make calls to Inference Providers** permission.
2. In the Vercel project connected to this repository, add the environment variable `HF_TOKEN` for Production (and Preview if needed). Never commit the token to GitHub or paste it into the website.
3. Redeploy the Vercel project after setting the variable.
4. Open `https://<your-vercel-domain>/health`. It should return JSON with `"model":"Ron-1 SmolLM"` and `"ready":true`.
5. In the Ron-1 GitHub Pages interface, open **إعداد اتصال النواة**, enter the Vercel domain, and test a real Arabic prompt.

## Optional environment variables

- `RON1_MODEL_ID`: defaults to `HuggingFaceTB/SmolLM3-3B`.
- `RON1_ALLOWED_ORIGIN`: defaults to `https://arkanws513-spec.github.io`.
- `RON1_MAX_NEW_TOKENS`: defaults to 256, maximum 512.
- `RON1_TEMPERATURE`: defaults to 0.7.
- `RON1_SYSTEM_PROMPT`: overrides Ron's system prompt.

The selected model must be available through a configured inference provider. Usage may consume free credits or require billing depending on account/provider. This proxy does not download or host model weights; Hugging Face's provider runs generation. Health only confirms configuration; a successful chat request is the actual generation test.

## Security

Keep `HF_TOKEN` only in Vercel environment variables. Do not add it to `index.html`, GitHub Pages, source control, or chat messages. Configure usage limits at the provider if available. Request size, history, and output length are bounded, but a public endpoint can still consume provider quota.
