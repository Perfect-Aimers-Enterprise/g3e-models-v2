import pytest

from g3e2.trainers.common import _aggregate_state_accuracy, save_checkpoint


class FakeModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = path


# ---------------------------------------------------------------------------
# _aggregate_state_accuracy — pure logic, no model/tensor needed
# ---------------------------------------------------------------------------
def test_perfect_predictions_give_100_percent():
    result = _aggregate_state_accuracy(["normal", "hazard"], ["normal", "hazard"])
    assert result["accuracy"] == 1.0
    assert result["correct"] == 2
    assert result["total"] == 2


def test_all_wrong_gives_zero_percent():
    result = _aggregate_state_accuracy(["normal", "normal"], ["hazard", "potential_threat"])
    assert result["accuracy"] == 0.0


def test_none_prediction_counts_as_wrong_not_excluded():
    result = _aggregate_state_accuracy([None, "normal"], ["normal", "normal"])
    assert result["total"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5


def test_per_class_accuracy_is_computed_independently():
    predicted = ["normal", "normal", "potential_threat"]
    expected = ["normal", "potential_threat", "potential_threat"]
    result = _aggregate_state_accuracy(predicted, expected)
    # normal: 1/1 correct; potential_threat: 1/2 correct
    assert result["per_class_accuracy"]["normal"] == 1.0
    assert result["per_class_accuracy"]["potential_threat"] == 0.5


def test_aggregate_reveals_hidden_rare_class_failure():
    """The exact scenario this function exists to catch: near-perfect
    aggregate accuracy hiding a model that never gets the rare class right."""
    predicted = ["normal"] * 96 + ["normal"] * 4  # never predicts potential_threat correctly
    expected = ["normal"] * 96 + ["potential_threat"] * 4
    result = _aggregate_state_accuracy(predicted, expected)
    assert result["accuracy"] == 0.96
    assert result["per_class_accuracy"]["potential_threat"] == 0.0


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        _aggregate_state_accuracy(["normal"], ["normal", "hazard"])


def test_empty_lists_return_zero_not_a_crash():
    result = _aggregate_state_accuracy([], [])
    assert result["accuracy"] == 0.0
    assert result["total"] == 0


# ---------------------------------------------------------------------------
# save_checkpoint — filesystem immutability guard, no torch needed
# ---------------------------------------------------------------------------
def _cfg():
    return {
        "versioning": {"model_name": "g3e-2", "version": "v1.0.0"},
        "model": {"base_model": "Qwen/Qwen2.5-VL-3B-Instruct"},
        "training": {"method": "lora", "learning_rate": 0.0002, "stages": ["a", "b"]},
        "lora": {"r": 16},
    }


def test_save_checkpoint_writes_version_info(tmp_path):
    model = FakeModel()
    out = save_checkpoint(model, tmp_path / "ckpt", _cfg())
    assert model.saved_to == out
    assert (out / "g3e_version_info.json").exists()


def test_save_checkpoint_refuses_to_overwrite_nonempty_dir(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "existing_file.txt").write_text("already here")

    with pytest.raises(RuntimeError, match="immutable"):
        save_checkpoint(FakeModel(), ckpt_dir, _cfg())


def test_save_checkpoint_allows_empty_existing_dir(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()  # exists but empty
    out = save_checkpoint(FakeModel(), ckpt_dir, _cfg())
    assert out == ckpt_dir


def test_save_checkpoint_excludes_stages_list_from_training_config():
    model = FakeModel()
    cfg = _cfg()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        out = save_checkpoint(model, Path(td) / "ckpt", cfg)
        import json
        info = json.loads((out / "g3e_version_info.json").read_text())
        assert "stages" not in info["training_config"]
        assert info["training_config"]["learning_rate"] == 0.0002
