import json
import urllib.request
from pathlib import Path

from core.config import settings

RELEASE_API = "https://api.github.com/repos/arkanws513-spec/Ron-1/releases/tags/qwen3-1.7b-weights-v1"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Ron-1/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def ensure_local_model() -> str:
    model_dir = Path(settings.model_id)
    if (model_dir / ".ron1-ready").exists():
        return str(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Ron-1/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)

    assets = {a["name"]: a["browser_download_url"] for a in release["assets"]}
    required = [
        "config.json", "generation_config.json", "merges.txt",
        "model-00001-of-00002.safetensors.part-00",
        "model-00001-of-00002.safetensors.part-01",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
    ]
    missing = [name for name in required if name not in assets]
    if missing:
        raise RuntimeError("Ron-1 model release is incomplete: " + ", ".join(missing))

    for name in required:
        target = model_dir / name
        if not target.exists():
            _download(assets[name], target)

    first = model_dir / "model-00001-of-00002.safetensors"
    if not first.exists():
        with first.open("wb") as out:
            for part_name in (
                "model-00001-of-00002.safetensors.part-00",
                "model-00001-of-00002.safetensors.part-01",
            ):
                with (model_dir / part_name).open("rb") as source:
                    while True:
                        chunk = source.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)

    (model_dir / ".ron1-ready").write_text(
        "Ron-1 local Qwen3-1.7B weights ready.\n", encoding="utf-8"
    )
    return str(model_dir)
