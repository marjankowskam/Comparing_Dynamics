"""Langevin equilibration followed by a full-batch inherent-structure quench."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..models import get_model
from ..optimizer import get_opt
from ..train_utils import get_task, init_weights, set_random_seed


@dataclass(frozen=True)
class InherentStructureConfig:
    """Numerical settings for one temperature and one trajectory.

    Langevin currently supports the SGD drift with isotropic noise. The quench
    optimizer can be any optimizer supported by the optimizer catalog.
    """

    temperature: float
    equilibration_steps: int
    quench_steps: int
    langevin_lr: float
    quench_lr: float
    langevin_optimizer_name: str = "sgd"
    langevin_optimizer_params: Optional[Dict[str, Any]] = None
    quench_optimizer_name: str = "sgd"
    quench_optimizer_params: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    quench_tolerance: float = 0.0


def _parameters_to_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def _optimizer_params(params: Optional[Dict[str, Any]], learning_rate: float) -> Dict[str, Any]:
    return dict(params) if params is not None else {"lr": learning_rate}


def _resolve_run_name(run_name: Optional[str]) -> str:
    if run_name:
        return run_name
    try:
        import wandb

        if wandb.run is not None and wandb.run.name:
            return wandb.run.name
    except ImportError:
        pass
    return "unnamed"


def _save_named_checkpoints(
    result: Dict[str, Any],
    save_dir: str | Path,
    run_name: Optional[str],
) -> Dict[str, str]:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in _resolve_run_name(run_name))

    equilibrated_path = output_dir / f"Equilibrated_{safe_name}.pt"
    inherent_path = output_dir / f"Inherent_{safe_name}.pt"
    result_path = output_dir / f"Result_{safe_name}.pt"
    torch.save(
        {
            "parameters": result["equilibrated_parameters"],
            "model_state_dict": result["equilibrated_model_state_dict"],
            "temperature": result["temperature"],
            "run_name": _resolve_run_name(run_name),
        },
        equilibrated_path,
    )
    torch.save(
        {
            "parameters": result["inherent_parameters"],
            "model_state_dict": result["model_state_dict"],
            "temperature": result["temperature"],
            "run_name": _resolve_run_name(run_name),
        },
        inherent_path,
    )
    torch.save(result, result_path)
    return {
        "equilibrated_checkpoint": str(equilibrated_path),
        "inherent_checkpoint": str(inherent_path),
        "result_file": str(result_path),
    }


def _full_batch_loss(model: torch.nn.Module, task: Any, data: Any, weights: Dict[str, float]) -> torch.Tensor:
    predictions = task.predict(data, model)
    return task.balanced_loss_fn(data, predictions, weights)


def _langevin_step(
    model: torch.nn.Module,
    task: Any,
    data: Any,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    weights: Dict[str, float],
    n_examples: int,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = _full_batch_loss(model, task, data, weights)
    loss.backward()
    optimizer.step()

    if len(optimizer.param_groups) != 1:
        raise ValueError("Langevin dynamics currently requires one optimizer parameter group.")
    learning_rate = optimizer.param_groups[0]["lr"]
    noise_scale = (2.0 * temperature * learning_rate / n_examples) ** 0.5
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * noise_scale)
    return float(loss.detach().cpu())


def _gd_step(
    model: torch.nn.Module,
    task: Any,
    data: Any,
    optimizer: torch.optim.Optimizer,
    weights: Dict[str, float],
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss = _full_batch_loss(model, task, data, weights)
    loss.backward()
    gradient_norm = torch.linalg.vector_norm(
        torch.cat([
            parameter.grad.detach().reshape(-1)
            for parameter in model.parameters()
            if parameter.grad is not None
        ])
    )
    optimizer.step()
    return float(loss.detach().cpu()), float(gradient_norm.cpu())


def cosine_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return q = <first, second> / (||first|| ||second||)."""

    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if denominator == 0:
        return 0.0
    return float(torch.dot(first, second).div(denominator).cpu())


def run_inherent_structure(
    cfg: Any,
    experiment_cfg: InherentStructureConfig,
    device: torch.device | str = "cpu",
    model: Optional[torch.nn.Module] = None,
    save_dir: Optional[str | Path] = None,
    run_name: Optional[str] = None,
    initial_parameters: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run one trajectory and return final metrics plus state snapshots.

    ``equilibrated_parameters`` is the state used as the initial condition for
    the quench. ``initial_parameters`` is the state before Langevin dynamics,
    or an explicitly supplied original initialization when ``model`` is a
    checkpoint loaded from a previous run.
    The reported ``overlap_q`` answers the quench-memory question by comparing
    the inherent structure to the equilibrated state.
    """

    if experiment_cfg.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if experiment_cfg.equilibration_steps < 0 or experiment_cfg.quench_steps < 0:
        raise ValueError("step counts must be non-negative")
    if experiment_cfg.seed is not None:
        set_random_seed(experiment_cfg.seed)

    device = torch.device(device)
    model_was_provided = model is not None
    model = model or get_model(cfg.model_name, cfg.model_params)
    model = model.to(device)
    if not model_was_provided:
        model.apply(init_weights)

    task = get_task(cfg.task_name, cfg.task_params, device)
    parameter_names = [name for name, _ in model.named_parameters()]
    task.initialize(parameter_names, model)
    full_data = task.get_full_data()
    loss_weights = task.initial_loss_weights()

    if initial_parameters is None:
        initial_parameters = _parameters_to_vector(model)
    else:
        initial_parameters = initial_parameters.detach().clone().to(device)
        if initial_parameters.numel() != sum(parameter.numel() for parameter in model.parameters()):
            raise ValueError("initial_parameters does not match the model parameter count")
    n_examples = len(full_data[0]) if isinstance(full_data, (tuple, list)) else 1

    langevin_params = _optimizer_params(
        experiment_cfg.langevin_optimizer_params,
        experiment_cfg.langevin_lr,
    )
    if experiment_cfg.langevin_optimizer_name != "sgd":
        raise NotImplementedError(
            "The isotropic Langevin noise implemented here is derived for SGD. "
            "Add an optimizer-specific noise covariance before using another "
            "Langevin optimizer."
        )
    langevin_optimizer = get_opt(
        experiment_cfg.langevin_optimizer_name,
        langevin_params,
        model.parameters(),
    )
    for _ in range(experiment_cfg.equilibration_steps):
        _langevin_step(
            model,
            task,
            full_data,
            langevin_optimizer,
            experiment_cfg.temperature,
            loss_weights,
            n_examples,
        )

    equilibrated_parameters = _parameters_to_vector(model)
    equilibrated_model_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    quench_optimizer = get_opt(
        experiment_cfg.quench_optimizer_name,
        _optimizer_params(experiment_cfg.quench_optimizer_params, experiment_cfg.quench_lr),
        model.parameters(),
    )
    final_loss = float("nan")
    final_gradient_norm = float("nan")
    for _ in range(experiment_cfg.quench_steps):
        final_loss, final_gradient_norm = _gd_step(model, task, full_data, quench_optimizer, loss_weights)
        if (
            experiment_cfg.quench_tolerance > 0
            and final_gradient_norm <= experiment_cfg.quench_tolerance
        ):
            break

    inherent_parameters = _parameters_to_vector(model)
    with torch.no_grad():
        final_loss = float(_full_batch_loss(model, task, full_data, loss_weights).cpu())

    result = {
        "temperature": experiment_cfg.temperature,
        "equilibration_steps": experiment_cfg.equilibration_steps,
        "quench_steps_requested": experiment_cfg.quench_steps,
        "final_loss": final_loss,
        "final_gradient_norm": final_gradient_norm,
        "overlap_q": cosine_overlap(equilibrated_parameters, inherent_parameters),
        "initial_to_inherent_overlap": cosine_overlap(initial_parameters, inherent_parameters),
        "initial_parameters": initial_parameters.cpu(),
        "equilibrated_parameters": equilibrated_parameters.cpu(),
        "inherent_parameters": inherent_parameters.cpu(),
        "equilibrated_model_state_dict": equilibrated_model_state_dict,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if save_dir is not None:
        result["saved_files"] = _save_named_checkpoints(result, save_dir, run_name)
    return result
