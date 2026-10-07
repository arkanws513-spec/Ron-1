from pathlib import Path

from huggingface_hub import snapshot_download

from core.config import settings


if __name__ == "__main__":
    destination = Path(settings.model_id)
    destination.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=settings.upstream_model_id,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )

    print(f"Ron-1 core model copied to: {path}")
    print("Base intelligence: Qwen3-1.7B")
