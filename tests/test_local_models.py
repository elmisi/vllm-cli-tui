"""Local disk scanning: extra dirs are pure filesystem work, testable offline."""
from vllm_tui.local_models import LocalModel, scan_extra_dirs


def test_each_subdirectory_is_one_model_with_recursive_size(tmp_path):
    (tmp_path / "my-model" / "sub").mkdir(parents=True)
    (tmp_path / "my-model" / "weights.bin").write_bytes(b"x" * 1000)
    (tmp_path / "my-model" / "sub" / "config.json").write_bytes(b"y" * 200)
    (tmp_path / "other-model").mkdir()
    (tmp_path / "other-model" / "weights.bin").write_bytes(b"z" * 500)
    (tmp_path / "loose-file.txt").write_bytes(b"ignored")

    models = scan_extra_dirs([str(tmp_path)])
    by_name = {m.name: m for m in models}
    assert by_name["my-model"].size_bytes == 1200
    assert by_name["other-model"].size_bytes == 500
    assert "loose-file.txt" not in by_name
    assert by_name["my-model"].source == str(tmp_path)


def test_missing_and_unreadable_dirs_are_skipped_quietly(tmp_path):
    assert scan_extra_dirs([str(tmp_path / "nope")]) == []
    assert scan_extra_dirs([]) == []


def test_results_are_sorted_by_size_descending(tmp_path):
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "f").write_bytes(b"x")
    (tmp_path / "big").mkdir()
    (tmp_path / "big" / "f").write_bytes(b"x" * 9000)
    models = scan_extra_dirs([str(tmp_path)])
    assert [m.name for m in models] == ["big", "small"]


def test_local_model_is_deletable_only_with_a_path():
    m = LocalModel(name="x", size_bytes=1, source="hf-cache", path="")
    assert not m.deletable
    m2 = LocalModel(name="x", size_bytes=1, source="/models", path="/models/x")
    assert m2.deletable
