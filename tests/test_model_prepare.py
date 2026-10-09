import hashlib
import io

import pytest

from model import prepare_model


def test_download_verifies_size_digest_and_writes_atomically(tmp_path, monkeypatch):
    payload = b"verified model asset"
    asset = {
        "browser_download_url": "https://example.invalid/model.bin",
        "size": len(payload),
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(
        prepare_model.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(payload),
    )
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    target = tmp_path / "model.bin"

    prepare_model._download(asset, target)

    assert target.read_bytes() == payload
    assert not (tmp_path / "model.bin.download").exists()


def test_download_rejects_corrupt_asset_and_cleans_temporary_file(tmp_path, monkeypatch):
    asset = {
        "browser_download_url": "https://example.invalid/model.bin",
        "size": 3,
        "digest": "sha256:" + hashlib.sha256(b"good").hexdigest(),
    }
    monkeypatch.setattr(
        prepare_model.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"bad"),
    )
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    target = tmp_path / "model.bin"

    with pytest.raises(RuntimeError, match="size mismatch|checksum mismatch"):
        prepare_model._download(asset, target)

    assert not target.exists()
    assert not (tmp_path / "model.bin.download").exists()


def test_github_headers_include_server_side_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-private-repo-token")
    headers = prepare_model._github_headers()
    assert headers["Authorization"] == "Bearer test-private-repo-token"
    assert headers["User-Agent"] == "Ron-1/1.0"


def test_authenticated_download_requires_api_asset_url(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "test-private-repo-token")
    asset = {
        "browser_download_url": "https://github.com/arkanws513-spec/Ron-1/releases/download/tag/model.bin",
        "size": 1,
    }
    with pytest.raises(RuntimeError, match="GitHub API asset URL"):
        prepare_model._download(asset, tmp_path / "model.bin")
