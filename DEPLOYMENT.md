# Ron-1 deployment

## Architecture
- Source code, chat UI, tests, and CI: GitHub.
- Static web UI: GitHub Pages (`index.html`).
- Chat API: FastAPI in `api/server.py`, deployable as a Vercel Python function (project root: `api`) or Docker service.
- Inference: Hugging Face Inference Providers using the OpenAI-compatible endpoint; the model weights are not downloaded into the API container.
- Default model: `Qwen/Qwen3-4B-Instruct-2507`; override with `HF_MODEL`.

## Required environment variables
- `HF_TOKEN`: a Hugging Face access token with permission to call inference providers. Store it only in the hosting provider's encrypted environment variables. Never place it in `index.html` or commit it to Git.
- `RON_ALLOWED_ORIGINS`: optional comma-separated list of allowed browser origins; defaults to `https://arkanws513-spec.github.io`.
- `RON_MAX_TOKENS`: optional output token limit (default 700).
- `RON_RATE_LIMIT_PER_MINUTE`: optional per-IP in-memory request limit (default 12).

## Deploy API
### Vercel
Set the project root directory to `api`, framework to FastAPI, and connect `arkanws513-spec/Ron-1`. Add `HF_TOKEN` in the project's encrypted environment variables, deploy, and test `/health` and `/chat`.

### Railway / Docker
Build from the repository root using `Dockerfile`, listen on the injected `PORT`, and add `HF_TOKEN` in service variables. Configure the public health check as `/health`.

After a public API URL is verified, set `API_BASE` in `index.html` to that exact URL and commit the change. The UI intentionally reports that the backend is not deployed while no verified API URL is configured.

## Important constraints
- The inference provider's free availability and rate limits can change; a valid token is required. Do not enable paid inference or paid hosting without the owner's approval.
- Railway currently rejected service creation because the account's free-plan resource limit was reached.
- Vercel project creation succeeded, but deployment was blocked by a Vercel account/scope authorization error. No paid plan was enabled and no payment was made.
- The local Qwen weights are not required by this remote-inference API; the original local Gradio app remains separate.