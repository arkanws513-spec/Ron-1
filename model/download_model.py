from huggingface_hub import snapshot_download
from core.config import settings


if __name__ == "__main__":
    path = snapshot_download(repo_id=settings.model_id)
    print(f"Ron-1 model ready: {path}")
