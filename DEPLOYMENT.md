# Ron-1 deployment

This repository is prepared to run Ron-1 as a normal Dockerized FastAPI service.

- Source: GitHub
- Model: Qwen3-1.7B
- Model weights: downloaded by Ron-1 from the GitHub Release at first startup
- API health: GET /health
- Chat API: POST /chat
- Container port: 8000

The model requires several GB of disk/RAM. Use a server with enough resources; a 1 GB free container is not sufficient for the full Qwen3-1.7B weights.
