from g3e_inference.artifacts import resolve_weights_file, resolve_adapter_dir


def test_resolve_weights_file_uses_local_path_when_it_exists(tmp_path, monkeypatch):
    local = tmp_path / "best.pt"
    local.write_bytes(b"fake weights")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called when a valid local path is given")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _should_not_be_called)

    result = resolve_weights_file(str(local), "some/repo", "best.pt")
    assert result == str(local)


def test_resolve_weights_file_falls_back_to_hf_when_local_missing(tmp_path, monkeypatch):
    calls = {}

    def _fake_download(repo_id, filename, token=None):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return "/fake/cached/best.pt"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_download)

    result = resolve_weights_file(str(tmp_path / "does_not_exist.pt"), "some/repo", "best.pt")
    assert result == "/fake/cached/best.pt"
    assert calls == {"repo_id": "some/repo", "filename": "best.pt"}


def test_resolve_weights_file_falls_back_to_hf_when_local_is_none(monkeypatch):
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: "/fake/path")
    result = resolve_weights_file(None, "some/repo", "best.pt")
    assert result == "/fake/path"


def test_resolve_weights_file_rejects_a_local_path_that_is_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: "/fake/from/hf")
    result = resolve_weights_file(str(tmp_path), "some/repo", "best.pt")
    assert result == "/fake/from/hf"


def test_resolve_adapter_dir_uses_local_directory_when_it_exists(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("snapshot_download should not be called when a valid local dir is given")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _should_not_be_called)

    result = resolve_adapter_dir(str(adapter_dir), "some/repo")
    assert result == str(adapter_dir)


def test_resolve_adapter_dir_falls_back_to_hf_when_local_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, token=None: "/fake/cached/adapter")
    result = resolve_adapter_dir(str(tmp_path / "nope"), "some/repo")
    assert result == "/fake/cached/adapter"


def test_resolve_adapter_dir_rejects_a_local_path_that_is_a_file(tmp_path, monkeypatch):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, token=None: "/fake/from/hf")
    result = resolve_adapter_dir(str(f), "some/repo")
    assert result == "/fake/from/hf"


def test_resolve_functions_pass_token_through(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, token=None: captured.update(token=token) or "/x",
    )
    resolve_weights_file(None, "r", "f", hf_token="secret123")
    assert captured["token"] == "secret123"
