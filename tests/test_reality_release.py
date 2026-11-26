import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from download_reality_release import install

    assets, checkout = tmp_path / "assets", tmp_path / "clone"
    assets.mkdir()
    files = []
    for name in ("head.pt", "config.json"):
        data = name.encode()
        (assets / name).write_bytes(data)
        files.append(dict(asset=name, path="runs/model/" + name,
                          sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    manifest = dict(repository="southern-star/keisim", tag="test", files=files)
    return assets, checkout, lambda: install(checkout, manifest, assets)


def test_installed_release_can_be_verified_offline(release):
    assets, checkout, install = release
    install()
    for asset in assets.iterdir():
        assert (checkout / "runs/model" / asset.name).read_bytes() == asset.read_bytes()
        asset.unlink()
    install()


def test_corrupt_download_leaves_no_partial_install(release):
    assets, checkout, install = release
    (assets / "config.json").write_bytes(b"x" * len("config.json"))
    with pytest.raises(ValueError, match="SHA-256"):
        install()
    assert not list((checkout / "runs/model").iterdir())


def test_existing_different_weights_are_preserved(release):
    _, checkout, install = release
    other = checkout / "runs/model/config.json"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"another model")
    with pytest.raises(ValueError, match="SHA-256"):
        install()
    assert other.read_bytes() == b"another model"
    assert not (other.parent / "head.pt").exists()


def test_background_release_upgrades_the_pinned_baseline(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from download_reality_release import CURRENT_TAG, MANIFEST

    baseline = json.loads(MANIFEST.read_text())["files"]
    current = json.loads(MANIFEST.with_name(CURRENT_TAG + ".json").read_text())
    assert current["tag"] == CURRENT_TAG and current["mode"] == "repair384"
    files = {item["path"]: item for item in current["files"]}
    assert len(files) == len({item["asset"] for item in files.values()}) == len(baseline) + 2
    for item in baseline:
        assert files.pop(item["path"]) == item
    assert set(files) == {"runs/reality/disocclusion384/fit/repair.pt", "runs/reality/disocclusion384/fit/selection.json"}
