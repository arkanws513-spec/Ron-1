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
    target = tmp_path / "model.bin"

    with pytest.raises(RuntimeError, match="size mismatch|checksum mismatch"):
        prepare_model._download(asset, target)

    assert not target.exists()
    assert not (tmp_path / "model.bin.download").exists()
