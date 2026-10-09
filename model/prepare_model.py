import hashlib
import json
import os
import urllib.request
from pathlib import Path

from core.config import settings

RELEASE_API = (
    "https://api.github.com/repos/arkanws513-spec/Ron-1/releases/"
    "tags/qwen3-1.7b-weights-v1"
)
CHUNK_SIZE = 8 * 1024 * 1024
PARTS = (
    "model-00001-of-00002.safetensors.part-00",
    "model-00001-of-00002.safetensors.part-01",
)
SMALL_ASSETS = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_file(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return False
    expected = asset.get("digest")
    if not expected:
        return True
    algorithm, separator, value = expected.partition(":")
    return bool(separator) and algorithm == "sha256" and _sha256(path) == value


def _download(asset: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        asset["browser_download_url"],
        headers={"User-Agent": "Ron-1/1.0"},
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())

        if size != asset["size"]:
            raise RuntimeError(
                f"Ron-1 asset size mismatch for {target.name}: "
                f"expected {asset['size']}, received {size}"
            )
        expected = asset.get("digest")
        if expected and expected.startswith("sha256:"):
            if digest.hexdigest() != expected.partition(":")[2]:
                raise RuntimeError(f"Ron-1 asset checksum mismatch for {target.name}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_asset(assets: dict[str, dict], name: str, target: Path) -> dict:
    asset = assets.get(name)
    if asset is None:
        raise RuntimeError(f"Ron-1 model release is missing required asset: {name}")
    if not _valid_file(target, asset):
        target.unlink(missing_ok=True)
        _download(asset, target)
    return asset


def ensure_local_model() -> str:
    model_dir = Path(settings.model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    first = model_dir / "model-00001-of-00002.safetensors"
    second = model_dir / "model-00002-of-00002.safetensors"
    marker = model_dir / ".ron1-ready"

    if marker.is_file() and all(
        (model_dir / name).is_file()
        for name in (*SMALL_ASSETS, first.name, second.name)
    ):
        return str(model_dir)

    request = urllib.request.Request(
        RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Ron-1/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)

    assets = {asset["name"]: asset for asset in release.get("assets", [])}

    # Download metadata first. Keep model weights out of the directory until
    # they are needed, which reduces peak disk use during first startup.
    for name in SMALL_ASSETS:
        _ensure_asset(assets, name, model_dir / name)

    expected_first_size = sum(assets.get(name, {}).get("size", 0) for name in PARTS)
    if not expected_first_size:
        raise RuntimeError("Ron-1 model release is missing the split first weight shard")

    if not _valid_file(first, {"size": expected_first_size}):
        first.unlink(missing_ok=True)
        for part_name in PARTS:
            part_path = model_dir / part_name
            asset = _ensure_asset(assets, part_name, part_path)
            with part_path.open("rb") as source, first.open("ab") as output:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            part_path.unlink(missing_ok=True)

        if first.stat().st_size != expected_first_size:
            first.unlink(missing_ok=True)
            raise RuntimeError("Ron-1 first weight shard reassembly failed")

    _ensure_asset(
        assets,
        second.name,
        second,
    )

    # The marker is written last; a failed or interrupted download is retried
    # safely on the next start instead of being mistaken for a ready model.
    temporary_marker = marker.with_name(marker.name + ".tmp")
    temporary_marker.write_text(
        "Ron-1 local Qwen3-1.7B weights verified.\n",
        encoding="utf-8",
    )
    os.replace(temporary_marker, marker)
    return str(model_dir)
