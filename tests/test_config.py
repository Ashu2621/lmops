"""Config loading, merging and validation."""

from __future__ import annotations

import pytest
import yaml

from lmops.config import Config, ModelConfig, load_config, save_config


def test_defaults_are_usable_without_a_file() -> None:
    config = load_config()
    assert isinstance(config, Config)
    assert config.model.n_layer > 0
    assert config.data.seq_len > 0


def test_yaml_overrides_defaults(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"model": {"n_layer": 3, "d_model": 128}, "train": {"max_steps": 9}}),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.n_layer == 3
    assert config.model.d_model == 128
    assert config.train.max_steps == 9
    # Untouched keys keep their defaults rather than being reset.
    assert config.optim.lr == pytest.approx(3e-4)


def test_dotted_overrides_win_over_yaml(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"train": {"max_steps": 100}}), encoding="utf-8")
    config = load_config(path, **{"train.max_steps": 5, "train.out_dir": "runs/x"})
    assert config.train.max_steps == 5
    assert config.train.out_dir == "runs/x"


def test_none_overrides_are_ignored(tmp_path) -> None:
    """The CLI passes ``None`` for flags the user did not set."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"train": {"max_steps": 42}}), encoding="utf-8")
    config = load_config(path, **{"train.max_steps": None})
    assert config.train.max_steps == 42


def test_unknown_keys_are_rejected_not_ignored(tmp_path) -> None:
    """A typo in a config file must fail loudly, not silently train the wrong model."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"optim": {"learning_rate": 0.1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(path)


def test_unknown_top_level_section_is_rejected(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"trainer": {"max_steps": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(path)


def test_non_mapping_yaml_is_rejected(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_save_and_reload_is_a_fixed_point(tmp_path) -> None:
    config = load_config(**{"model.n_layer": 4, "train.seed": 99})
    path = tmp_path / "out.yaml"
    save_config(config, path)
    assert load_config(path).to_dict() == config.to_dict()


def test_shipped_configs_all_load() -> None:
    """Every YAML under configs/ must parse -- these are the entry points a reader tries."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(root.glob("*.yaml"))
    assert files, "no configs found"
    for path in files:
        config = load_config(path)
        assert config.model.d_model > 0, path


def test_head_dim_and_repetition_are_derived() -> None:
    config = ModelConfig(d_model=384, n_head=6, n_kv_head=2)
    assert config.head_dim == 64
    assert config.n_rep == 3


def test_n_kv_head_defaults_to_mha() -> None:
    config = ModelConfig(d_model=64, n_head=4)
    assert config.n_kv_head == 4
    assert config.n_rep == 1
