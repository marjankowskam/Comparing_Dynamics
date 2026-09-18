"""Shared interfaces for experiments that run one or more training trajectories."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MetricsConfig:
    """Controls which metrics are retained during a trajectory."""

    save_every_step: bool = True
    metric_names: Optional[Iterable[str]] = None


@dataclass
class TrajectoryResult:
    """Metrics and metadata produced by one training trajectory."""

    trajectory_id: str
    final_metrics: Dict[str, Any]
    history: Dict[str, List[Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Experiment:
    """Base class for studies composed of multiple independent trajectories."""

    def __init__(self, metrics: Optional[MetricsConfig] = None):
        self.metrics = metrics or MetricsConfig()

    def run_trajectory(self, trajectory_id: str, **kwargs: Any) -> TrajectoryResult:
        raise NotImplementedError

    def run(self, trajectory_ids: Iterable[str], **kwargs: Any) -> List[TrajectoryResult]:
        return [
            self.run_trajectory(trajectory_id, **kwargs)
            for trajectory_id in trajectory_ids
        ]


def record_metrics(
    history: Dict[str, List[Any]],
    metrics: Dict[str, Any],
    config: MetricsConfig,
) -> None:
    """Append selected metrics to history according to the experiment policy."""

    if not config.save_every_step:
        return

    selected_names = set(config.metric_names) if config.metric_names is not None else None
    for name, value in metrics.items():
        if selected_names is None or name in selected_names:
            history.setdefault(name, []).append(value)
