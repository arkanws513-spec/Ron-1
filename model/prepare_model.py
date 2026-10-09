import hashlib
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

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


class _StripCrossHostAuthorization(urllib.request.HTTPRedirectHandler):
    """Never forward the GitHub token to the signed release-asset CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlparse(req.full_url).hostname != urlparse(newurl).hostname:
            for header in ("Authorization", "authorization", "Proxy-Authorization"):
                redirected.remove_header(header)
                redirected.remove_unredirected_header(header)
        return redirected


def _github_token() -> str:
    return os.getenv("GITHUB_TOKEN", "").strip()


def _github_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {"Accept": accept, "User-Agent": "Ron-1/1.0"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _open_url(request: urllib.request.Request, timeout: int):
    # Keep urlopen as the simple public path (and test seam). When a token is
    # present, use a redirect handler that strips credentials on host changes.
    if not _github_token():
        return urllib.request.urlopen(request, timeout=timeout)
    opener = urllib.request.build_opener(_StripCrossHostAuthorization())
    return opener.open(request, timeout=timeout)


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
    # Private release assets require the authenticated GitHub API asset URL.
    # The API redirects to a signed CDN URL; authorization is stripped when
    # the redirect crosses hosts.
    download_url = asset.get("url") or asset.get("browser_download_url")
    if not download_url:
        raise RuntimeError(f"Ron-1 release asset has no download URL: {target.name}")
    if _github_token() and urlparse(download_url).hostname == "github.com":
        raise RuntimeError(
            "Authenticated model downloads must use the GitHub API asset URL; "
            "the release metadata did not include one."
        )
    request = urllib.request.Request(
        download_url,
        headers=_github_headers("application/octet-stream"),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with _open_url(request, timeout=120) as response:
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
    except Exception as error:
        temporary.unlink(missing_ok=True)
        if getattr(error, "code", None) == 404 and not _github_token():
            raise RuntimeError(
                "Ron-1 model release was not accessible. If the repository is private, "
                "configure GITHUB_TOKEN with read access to repository contents and release assets."
            ) from error
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

    request = urllib.request.Request(RELEASE_API, headers=_github_headers())
    try:
        with _open_url(request, timeout=30) as response:
            release = json.load(response)
    except Exception as error:
        if getattr(error, "code", None) == 404 and not _github_token():
            raise RuntimeError(
                "Ron-1 model release was not accessible. If the repository is private, "
                "configure GITHUB_TOKEN with read access to repository contents and release assets."
            ) from error
        raise

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
            _ensure_asset(assets, part_name, part_path)
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

    _ensure_asset(assets, second.name, model_dir / second.name)

    # The marker is written last; a failed or interrupted download is retried
    # safely on the next start instead of being mistaken for a ready model.
    temporary_marker = marker.with_name(marker.name + ".tmp")
    temporary_marker.write_text(
        "Ron-1 local Qwen3-1.7B weights verified.\n",
        encoding="utf-8",
    )
    os.replace(temporary_marker, marker)
    return str(model_dir)
