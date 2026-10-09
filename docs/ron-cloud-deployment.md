# Ron-1 on Ron Cloud

## Service contract

Ron-1 is a separate FastAPI inference service. Ron Cloud must call it server-to-server; do not put `RON_API_KEY` in browser code or expose it to end users.

- Container port: `8000`
- Liveness: `GET /health` (does not load the model)
- Chat: `POST /chat`
- Required secret: `RON_API_KEY`
- Persistent mount: `/data`
- Model directory: `/data/models/Ron-1-Qwen3-1.7B`
- Conversation memory root: `/data/memory`

Example request from the authenticated Ron Cloud backend:

```http
POST /chat
Content-Type: application/json
X-Ron-API-Key: <server-side secret>

{
  "message": "مرحبا يا رون",
  "user_id": "<verified Ron Cloud account ID>",
  "conversation_id": "<conversation ID>"
}
```

The `user_id` must come from Ron Cloud's verified session, never from a browser-supplied identity claim. Keep the API private to Ron Cloud where the host supports private networking. If public networking is required, retain the API key and add platform-level rate limiting.

## Runtime requirements

The Qwen3-1.7B weights are several gigabytes. Do not deploy this service on a 0.5 GB RAM / 0.5 GB disk free container. Plan for at least 8 GB RAM and a persistent volume of 10 GB to leave room for model downloads, reassembly, and logs. CPU-only inference is supported but slower than a suitable GPU.

The Docker image installs CPU PyTorch and runs as a non-root user. Mount a persistent volume at `/data`; without it, model downloads and conversation memory are lost on replacement of the container.

## Environment

Set these server-side variables:

- `RON_API_KEY`: long random secret shared only between Ron Cloud's backend and this service.
- `RON_MODEL_ID=/data/models/Ron-1-Qwen3-1.7B`
- `RON_MEMORY_PATH=/data/memory`
- `PORT=8000`

Optional generation settings are `RON_MAX_NEW_TOKENS`, `RON_TEMPERATURE`, and `RON_TOP_P`.

## Deployment sequence

1. Build the Docker image and run `GET /health`.
2. Attach persistent storage at `/data` and set the variables above.
3. Verify an unauthenticated chat receives `401`, a missing server key receives `503`, and an authenticated chat succeeds.
4. Verify two different `user_id` values cannot see one another's conversation history.
5. Configure Ron Cloud's backend to call the private Ron-1 service with the server-side key and verified user/conversation IDs.
6. Run a real inference and restart the service to confirm the model cache and memory persist.

A successful image build is not proof of a live deployment. Mark Ron-1 as hosted only after the real service passes health, authenticated inference, persistence, and isolation checks.
