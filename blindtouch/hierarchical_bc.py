"""Trajectory storage and hierarchical behavior cloning for BlindTouch.

The trajectory store preserves skill labels and teacher metadata so offline
demonstrations do not collapse into one anonymous action-regression soup.  The
hierarchical policy still exposes the same ``predict`` method used by the rest
of the training/evaluation code: observations in, actions out.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .objects import TOUCH_SKILL_MODES


Observation = NDArray[np.float32]
Action = NDArray[np.float32]
ModeIdArray = NDArray[np.int64]
MODE_NAMES = tuple(TOUCH_SKILL_MODES)
MODE_TO_ID = {mode: index for index, mode in enumerate(MODE_NAMES)}
OBSERVATION_SIZE = 360
ACTION_SIZE = 4


@dataclass(frozen=True)
class TrajectoryStore:
    """Flat transition arrays plus episode-level metadata."""

    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    mode_ids: NDArray[np.int64]
    episode_ids: NDArray[np.int64]
    step_ids: NDArray[np.int64]
    branches: tuple[str, ...]
    episode_metadata: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        mode_ids = np.asarray(self.mode_ids, dtype=np.int64)
        episode_ids = np.asarray(self.episode_ids, dtype=np.int64)
        step_ids = np.asarray(self.step_ids, dtype=np.int64)
        branches = tuple(str(branch) for branch in self.branches)
        episode_metadata = tuple(dict(row) for row in self.episode_metadata)
        metadata = dict(self.metadata or {})

        transition_count = int(observations.shape[0])
        if observations.ndim != 2 or observations.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"observations must have shape (N, {OBSERVATION_SIZE}), got {observations.shape}"
            )
        if actions.shape != (transition_count, ACTION_SIZE):
            raise ValueError(f"actions must have shape (N, {ACTION_SIZE}), got {actions.shape}")
        for name, array in (
            ("mode_ids", mode_ids),
            ("episode_ids", episode_ids),
            ("step_ids", step_ids),
        ):
            if array.shape != (transition_count,):
                raise ValueError(f"{name} must have shape (N,), got {array.shape}")
        if len(branches) != transition_count:
            raise ValueError("branches must contain one label per transition")
        if mode_ids.size and (np.min(mode_ids) < 0 or np.max(mode_ids) >= len(MODE_NAMES)):
            raise ValueError("mode_ids contain an unsupported mode index")

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "mode_ids", mode_ids)
        object.__setattr__(self, "episode_ids", episode_ids)
        object.__setattr__(self, "step_ids", step_ids)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "episode_metadata", episode_metadata)
        object.__setattr__(self, "metadata", metadata)

    @property
    def transition_count(self) -> int:
        return int(self.observations.shape[0])

    @property
    def episode_count(self) -> int:
        if self.episode_metadata:
            return len(self.episode_metadata)
        return len(set(int(episode_id) for episode_id in self.episode_ids.tolist()))

    def summary(self) -> dict[str, Any]:
        mode_counts = {
            MODE_NAMES[mode_id]: int(np.count_nonzero(self.mode_ids == mode_id))
            for mode_id in range(len(MODE_NAMES))
        }
        branch_counts: dict[str, int] = {}
        for branch in self.branches:
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
        return {
            "transitions": self.transition_count,
            "episodes": self.episode_count,
            "mode_counts": mode_counts,
            "branch_counts": dict(sorted(branch_counts.items())),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            observations=self.observations,
            actions=self.actions,
            mode_ids=self.mode_ids,
            episode_ids=self.episode_ids,
            step_ids=self.step_ids,
            branches=np.asarray(self.branches, dtype=np.str_),
            episode_metadata_json=np.asarray(
                json.dumps(list(self.episode_metadata), sort_keys=True),
                dtype=np.str_,
            ),
            metadata_json=np.asarray(
                json.dumps(self.metadata or {}, sort_keys=True),
                dtype=np.str_,
            ),
            mode_names=np.asarray(MODE_NAMES, dtype=np.str_),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryStore":
        with np.load(Path(path), allow_pickle=False) as data:
            mode_names = tuple(str(name) for name in data["mode_names"].tolist())
            if mode_names != MODE_NAMES:
                raise ValueError(
                    f"Trajectory store modes {mode_names!r} do not match {MODE_NAMES!r}"
                )
            return cls(
                observations=data["observations"],
                actions=data["actions"],
                mode_ids=data["mode_ids"],
                episode_ids=data["episode_ids"],
                step_ids=data["step_ids"],
                branches=tuple(str(branch) for branch in data["branches"].tolist()),
                episode_metadata=tuple(
                    json.loads(str(data["episode_metadata_json"].item()))
                ),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def concatenate_stores(stores: Sequence[TrajectoryStore]) -> TrajectoryStore:
    """Merge compatible stores while keeping episode ids unique."""

    if not stores:
        raise ValueError("stores must contain at least one TrajectoryStore")
    observations: list[NDArray[np.float32]] = []
    actions: list[NDArray[np.float32]] = []
    mode_ids: list[NDArray[np.int64]] = []
    episode_ids: list[NDArray[np.int64]] = []
    step_ids: list[NDArray[np.int64]] = []
    branches: list[str] = []
    episode_metadata: list[dict[str, Any]] = []
    episode_offset = 0
    for store in stores:
        observations.append(store.observations)
        actions.append(store.actions)
        mode_ids.append(store.mode_ids)
        episode_ids.append(store.episode_ids + episode_offset)
        step_ids.append(store.step_ids)
        branches.extend(store.branches)
        for metadata in store.episode_metadata:
            row = dict(metadata)
            row["episode_id"] = int(row.get("episode_id", 0)) + episode_offset
            episode_metadata.append(row)
        episode_offset += store.episode_count
    return TrajectoryStore(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        mode_ids=np.concatenate(mode_ids, axis=0),
        episode_ids=np.concatenate(episode_ids, axis=0),
        step_ids=np.concatenate(step_ids, axis=0),
        branches=tuple(branches),
        episode_metadata=tuple(episode_metadata),
        metadata={"source_count": len(stores)},
    )


@dataclass(frozen=True)
class HierarchicalBCConfig:
    """Supervised training settings for the mode-conditioned actor."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 3e-4
    validation_fraction: float = 0.10
    mode_loss_weight: float = 0.25
    action_loss_weight: float = 1.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(width < 1 for width in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer widths")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.mode_loss_weight < 0.0 or self.action_loss_weight <= 0.0:
            raise ValueError("loss weights must be nonnegative, with positive action weight")


@dataclass(frozen=True)
class HierarchicalBCResult:
    """Training result for a hierarchical behavior-cloned policy."""

    policy: "HierarchicalBCPolicy"
    metrics: dict[str, Any]


class HierarchicalBCPolicy:
    """Predictive wrapper around a torch mode head and action heads."""

    def __init__(
        self,
        config: HierarchicalBCConfig,
        *,
        observation_mean: NDArray[np.float32] | None = None,
        observation_std: NDArray[np.float32] | None = None,
    ) -> None:
        self.config = config
        self._torch = _load_torch()
        self.device = _select_device(self._torch, config.device)
        self.network = _build_network(self._torch, config).to(self.device)
        self.observation_mean = _observation_stat_or_default(observation_mean, default=0.0)
        self.observation_std = _observation_stat_or_default(observation_std, default=1.0)
        self._observation_mean_tensor = self._torch.as_tensor(
            self.observation_mean, dtype=self._torch.float32, device=self.device
        )
        self._observation_std_tensor = self._torch.as_tensor(
            self.observation_std, dtype=self._torch.float32, device=self.device
        )

    def predict(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> tuple[NDArray[np.float32], None]:
        del deterministic
        single_observation = np.asarray(observation, dtype=np.float32)
        batched = single_observation.ndim == 2
        if single_observation.ndim == 1:
            single_observation = single_observation[None, :]
        if single_observation.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"Expected observation size {OBSERVATION_SIZE}, got {single_observation.shape}"
            )
        self.network.eval()
        with self._torch.no_grad():
            obs_tensor = self._torch.as_tensor(
                single_observation, dtype=self._torch.float32, device=self.device
            )
            obs_tensor = self.normalize_tensor(obs_tensor)
            mode_logits, action_heads = self.network(obs_tensor)
            mode_ids = self._torch.argmax(mode_logits, dim=1)
            batch_indices = self._torch.arange(
                obs_tensor.shape[0], dtype=self._torch.long, device=self.device
            )
            actions = action_heads[batch_indices, mode_ids].detach().cpu().numpy()
        actions = np.clip(actions.astype(np.float32), -1.0, 1.0)
        return (actions if batched else actions[0]), None

    def predict_mode(self, observation: NDArray[np.float32]) -> str:
        single_observation = np.asarray(observation, dtype=np.float32)
        if single_observation.ndim != 1:
            raise ValueError("predict_mode expects one flattened observation")
        self.network.eval()
        with self._torch.no_grad():
            obs_tensor = self._torch.as_tensor(
                single_observation[None, :], dtype=self._torch.float32, device=self.device
            )
            obs_tensor = self.normalize_tensor(obs_tensor)
            mode_logits, _ = self.network(obs_tensor)
            mode_id = int(self._torch.argmax(mode_logits, dim=1).item())
        return MODE_NAMES[mode_id]

    def normalize_tensor(self, observations: Any) -> Any:
        return (observations - self._observation_mean_tensor) / self._observation_std_tensor


def train_hierarchical_bc(
    store: TrajectoryStore, config: HierarchicalBCConfig
) -> HierarchicalBCResult:
    """Train a mode classifier plus mode-conditioned action heads."""

    if store.transition_count < 1:
        raise ValueError("Trajectory store is empty")
    torch = _load_torch()
    _seed_torch(torch, config.seed)
    rng = np.random.default_rng(config.seed)
    train_indices, validation_indices = _split_indices(
        rng, store.transition_count, config.validation_fraction
    )
    observation_mean, observation_std = _fit_observation_stats(store.observations[train_indices])
    policy = HierarchicalBCPolicy(
        config,
        observation_mean=observation_mean,
        observation_std=observation_std,
    )
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=config.learning_rate)
    obs_tensor = torch.as_tensor(store.observations, dtype=torch.float32, device=policy.device)
    action_tensor = torch.as_tensor(store.actions, dtype=torch.float32, device=policy.device)
    mode_tensor = torch.as_tensor(store.mode_ids, dtype=torch.long, device=policy.device)
    final_train: dict[str, float] = {}
    final_validation: dict[str, float] = {}

    for _ in range(config.epochs):
        policy.network.train()
        for batch in _batch_indices(rng, train_indices, config.batch_size):
            losses = _losses_for_batch(
                torch,
                policy,
                config,
                obs_tensor[batch],
                action_tensor[batch],
                mode_tensor[batch],
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
        final_train = _evaluate_loss_split(
            torch, policy, config, obs_tensor, action_tensor, mode_tensor, train_indices
        )
        if validation_indices.size:
            final_validation = _evaluate_loss_split(
                torch,
                policy,
                config,
                obs_tensor,
                action_tensor,
                mode_tensor,
                validation_indices,
            )

    metrics = {
        "train": final_train,
        "validation": final_validation,
        "store": store.summary(),
        "config": asdict(config),
    }
    return HierarchicalBCResult(policy=policy, metrics=metrics)


def save_hierarchical_policy(
    policy: HierarchicalBCPolicy,
    path: str | Path,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Save a trained hierarchical policy checkpoint."""

    torch = _load_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(policy.config),
            "state_dict": policy.network.state_dict(),
            "observation_mean": policy.observation_mean.tolist(),
            "observation_std": policy.observation_std.tolist(),
            "mode_names": MODE_NAMES,
            "metrics": dict(metrics or {}),
        },
        destination,
    )
    return destination


def load_hierarchical_policy(
    path: str | Path, *, device: str = "auto"
) -> tuple[HierarchicalBCPolicy, dict[str, Any]]:
    """Load a policy checkpoint created by :func:`save_hierarchical_policy`."""

    torch = _load_torch()
    selected_device = _select_device(torch, device)
    checkpoint = torch.load(Path(path), map_location=selected_device)
    mode_names = tuple(checkpoint["mode_names"])
    if mode_names != MODE_NAMES:
        raise ValueError(f"Checkpoint modes {mode_names!r} do not match {MODE_NAMES!r}")
    config_values = dict(checkpoint["config"])
    config_values["device"] = device
    config = HierarchicalBCConfig(**config_values)
    policy = HierarchicalBCPolicy(
        config,
        observation_mean=checkpoint.get("observation_mean"),
        observation_std=checkpoint.get("observation_std"),
    )
    policy.network.load_state_dict(checkpoint["state_dict"])
    policy.network.eval()
    return policy, dict(checkpoint.get("metrics") or {})


def _losses_for_batch(
    torch: Any,
    policy: HierarchicalBCPolicy,
    config: HierarchicalBCConfig,
    observations: Any,
    actions: Any,
    mode_ids: Any,
) -> dict[str, Any]:
    import torch.nn.functional as functional

    normalized_observations = policy.normalize_tensor(observations)
    mode_logits, action_heads = policy.network(normalized_observations)
    batch_indices = torch.arange(observations.shape[0], dtype=torch.long, device=policy.device)
    selected_actions = action_heads[batch_indices, mode_ids]
    mode_loss = functional.cross_entropy(mode_logits, mode_ids)
    action_loss = functional.mse_loss(selected_actions, actions)
    loss = config.action_loss_weight * action_loss + config.mode_loss_weight * mode_loss
    mode_accuracy = (torch.argmax(mode_logits, dim=1) == mode_ids).float().mean()
    return {
        "loss": loss,
        "action_loss": action_loss,
        "mode_loss": mode_loss,
        "mode_accuracy": mode_accuracy,
    }


def _evaluate_loss_split(
    torch: Any,
    policy: HierarchicalBCPolicy,
    config: HierarchicalBCConfig,
    obs_tensor: Any,
    action_tensor: Any,
    mode_tensor: Any,
    indices: NDArray[np.int64],
) -> dict[str, float]:
    policy.network.eval()
    with torch.no_grad():
        losses = _losses_for_batch(
            torch,
            policy,
            config,
            obs_tensor[indices],
            action_tensor[indices],
            mode_tensor[indices],
        )
    return {
        key: float(value.detach().cpu().item())
        for key, value in losses.items()
    }


def _split_indices(
    rng: np.random.Generator, count: int, validation_fraction: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    indices = rng.permutation(count).astype(np.int64)
    validation_count = int(round(count * validation_fraction))
    if validation_count >= count:
        validation_count = count - 1
    validation = indices[:validation_count]
    train = indices[validation_count:]
    return train, validation


def _batch_indices(
    rng: np.random.Generator, indices: NDArray[np.int64], batch_size: int
) -> Iterable[NDArray[np.int64]]:
    permutation = rng.permutation(indices)
    for start in range(0, len(permutation), batch_size):
        yield permutation[start : start + batch_size]


def _build_network(torch: Any, config: HierarchicalBCConfig) -> Any:
    layers: list[Any] = []
    input_width = OBSERVATION_SIZE
    for width in config.hidden_sizes:
        layers.append(torch.nn.Linear(input_width, width))
        layers.append(torch.nn.ReLU())
        input_width = width

    class _Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(*layers)
            self.mode_head = torch.nn.Linear(input_width, len(MODE_NAMES))
            self.action_heads = torch.nn.ModuleList(
                torch.nn.Linear(input_width, ACTION_SIZE) for _ in MODE_NAMES
            )

        def forward(self, observations: Any) -> tuple[Any, Any]:
            features = self.encoder(observations)
            mode_logits = self.mode_head(features)
            actions = torch.stack(
                [torch.tanh(head(features)) for head in self.action_heads],
                dim=1,
            )
            return mode_logits, actions

    return _Network()


def _fit_observation_stats(
    observations: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    mean = np.asarray(np.mean(observations, axis=0), dtype=np.float32)
    std = np.asarray(np.std(observations, axis=0), dtype=np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def _observation_stat_or_default(
    value: NDArray[np.float32] | None, *, default: float
) -> NDArray[np.float32]:
    if value is None:
        return np.full((OBSERVATION_SIZE,), default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"observation statistic must have shape ({OBSERVATION_SIZE},)")
    return array


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Hierarchical BC requires PyTorch") from error
    return torch


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("store", type=Path)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("store", type=Path)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--validation-fraction", type=float, default=0.10)
    train_parser.add_argument("--mode-loss-weight", type=float, default=0.25)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    if args.command == "summary":
        print(json.dumps(TrajectoryStore.load(args.store).summary(), indent=2, sort_keys=True))
        return

    store = TrajectoryStore.load(args.store)
    config = HierarchicalBCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        mode_loss_weight=args.mode_loss_weight,
        seed=args.seed,
        device=args.device,
    )
    result = train_hierarchical_bc(store, config)
    save_hierarchical_policy(result.policy, args.output, metrics=result.metrics)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ACTION_SIZE",
    "MODE_NAMES",
    "MODE_TO_ID",
    "OBSERVATION_SIZE",
    "HierarchicalBCConfig",
    "HierarchicalBCPolicy",
    "HierarchicalBCResult",
    "TrajectoryStore",
    "concatenate_stores",
    "load_hierarchical_policy",
    "save_hierarchical_policy",
    "train_hierarchical_bc",
]
