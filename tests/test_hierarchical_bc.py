import json

import numpy as np
import pytest

from blindtouch.hierarchical_bc import (
    ACTION_SIZE,
    MODE_NAMES,
    MODE_TO_ID,
    OBSERVATION_SIZE,
    HierarchicalBCConfig,
    TrajectoryStore,
    concatenate_stores,
    load_hierarchical_policy,
    save_hierarchical_policy,
    train_hierarchical_bc,
)


def _tiny_store(count: int = 12) -> TrajectoryStore:
    observations = np.zeros((count, OBSERVATION_SIZE), dtype=np.float32)
    actions = np.zeros((count, ACTION_SIZE), dtype=np.float32)
    mode_ids = np.arange(count, dtype=np.int64) % len(MODE_NAMES)
    episode_ids = np.arange(count, dtype=np.int64) // 3
    step_ids = np.arange(count, dtype=np.int64) % 3
    for row, mode_id in enumerate(mode_ids):
        observations[row, mode_id] = 1.0
        actions[row] = np.array(
            [mode_id / 10.0, 0.01 * row, -0.01 * row, 0.02],
            dtype=np.float32,
        )
    return TrajectoryStore(
        observations=observations,
        actions=actions,
        mode_ids=mode_ids,
        episode_ids=episode_ids,
        step_ids=step_ids,
        branches=tuple(f"branch_{int(mode_id)}" for mode_id in mode_ids),
        episode_metadata=tuple(
            {"episode_id": index, "outcome": "success"}
            for index in sorted(set(episode_ids.tolist()))
        ),
        metadata={"source": "unit"},
    )


def test_trajectory_store_round_trips_npz_with_mode_metadata(tmp_path) -> None:
    store = _tiny_store()
    path = store.save(tmp_path / "store.npz")

    loaded = TrajectoryStore.load(path)

    np.testing.assert_allclose(loaded.observations, store.observations)
    np.testing.assert_allclose(loaded.actions, store.actions)
    np.testing.assert_array_equal(loaded.mode_ids, store.mode_ids)
    assert loaded.branches == store.branches
    assert loaded.metadata == {"source": "unit"}
    assert loaded.summary()["mode_counts"]["round_retention"] == 3
    assert loaded.summary()["branch_counts"]["branch_0"] == 3


def test_trajectory_store_validates_shapes_and_modes() -> None:
    store = _tiny_store()
    with pytest.raises(ValueError, match="observations"):
        TrajectoryStore(
            observations=np.zeros((2, 45), dtype=np.float32),
            actions=np.zeros((2, ACTION_SIZE), dtype=np.float32),
            mode_ids=np.zeros(2, dtype=np.int64),
            episode_ids=np.zeros(2, dtype=np.int64),
            step_ids=np.zeros(2, dtype=np.int64),
            branches=("a", "b"),
        )
    with pytest.raises(ValueError, match="unsupported mode"):
        TrajectoryStore(
            observations=store.observations,
            actions=store.actions,
            mode_ids=np.full(store.transition_count, len(MODE_NAMES), dtype=np.int64),
            episode_ids=store.episode_ids,
            step_ids=store.step_ids,
            branches=store.branches,
        )


def test_concatenate_stores_offsets_episode_ids() -> None:
    merged = concatenate_stores((_tiny_store(6), _tiny_store(6)))

    assert merged.transition_count == 12
    assert merged.episode_count == 4
    assert int(np.max(merged.episode_ids)) == 3
    assert merged.metadata == {"source_count": 2}


def test_hierarchical_bc_trains_mode_head_and_action_heads(tmp_path) -> None:
    pytest.importorskip("torch")
    store = _tiny_store(40)
    config = HierarchicalBCConfig(
        hidden_sizes=(32,),
        epochs=4,
        batch_size=8,
        learning_rate=1e-2,
        validation_fraction=0.2,
        seed=3,
        device="cpu",
    )

    result = train_hierarchical_bc(store, config)
    action, _ = result.policy.predict(store.observations[0])
    mode = result.policy.predict_mode(store.observations[0])
    checkpoint = save_hierarchical_policy(result.policy, tmp_path / "hbc.pt", metrics=result.metrics)
    loaded, metrics = load_hierarchical_policy(checkpoint, device="cpu")
    loaded_action, _ = loaded.predict(store.observations[0])

    assert action.shape == (ACTION_SIZE,)
    assert mode in MODE_NAMES
    assert result.metrics["train"]["mode_accuracy"] >= 0.0
    assert metrics["store"]["transitions"] == 40
    np.testing.assert_allclose(loaded_action, action, atol=1e-6)


def test_hierarchical_bc_can_weight_tactile_contact_states() -> None:
    pytest.importorskip("torch")
    store = _tiny_store(40)
    fragile_id = MODE_TO_ID["fragile_balance"]
    fragile_rows = store.mode_ids == fragile_id
    store.observations[fragile_rows, 12:39] = 1.0
    config = HierarchicalBCConfig(
        hidden_sizes=(32,),
        epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        validation_fraction=0.0,
        mode_loss_weight=0.0,
        tactile_action_weight=2.0,
        tactile_action_weight_mode="fragile_balance",
        seed=4,
        device="cpu",
    )

    result = train_hierarchical_bc(store, config)

    assert result.metrics["train"]["mean_action_weight"] > 1.0
    assert result.metrics["train"]["weighted_action_loss"] >= result.metrics["train"]["action_loss"]


def test_hierarchical_bc_cli_summary(tmp_path, capsys) -> None:
    from blindtouch import hierarchical_bc

    path = _tiny_store().save(tmp_path / "store.npz")

    hierarchical_bc.main(["summary", str(path)])

    output = json.loads(capsys.readouterr().out)
    assert output["transitions"] == 12
    assert output["metadata"] == {"source": "unit"}
