"""Langevin equilibration followed by a full-batch inherent-structure quench."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import warnings

from ..models import get_model
from ..optimizer import get_opt
from ..train_utils import get_task, init_weights, set_random_seed


@dataclass(frozen=True)
class InherentStructureConfig:
    """Numerical settings for one temperature and one trajectory.

    Langevin currently supports the SGD drift with isotropic noise. The quench
    optimizer can be any optimizer supported by the optimizer catalog.
    """
    equilibration_steps: int
    quench_steps: int

    equilibration_optimizer_name: str = "langevin"
    equilibration_optimizer_params: Optional[Dict[str, Any]] = None

    quench_optimizer_name: str = "sgd"
    quench_optimizer_params: Optional[Dict[str, Any]] = None
    quench_tolerance: float = 0.0

    model_seed: Optional[int] = None #seed to generate the model weights, if model is not provided
    seed: Optional[int] = None #seed for all the randomness in the dynamics, used as model seed if model_seed is not provided
    
    wandb_log_every: int = 0
    metrics: tuple[str, ...] = ()


def _parameters_to_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

# def _optimizer_params(params: Optional[Dict[str, Any]], learning_rate: float) -> Dict[str, Any]:
#     return dict(params) if params is not None else {"lr": learning_rate}


def _wandb_log(metrics: Dict[str, Any], step: int) -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


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


def _save_named_checkpoints(result: Dict[str, Any],  save_dir: str | Path, run_name: Optional[str]) -> Dict[str, str]:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in _resolve_run_name(run_name))

    equilibrated_path = output_dir / f"Equilibrated_{safe_name}.pt"
    inherent_path = output_dir / f"Inherent_{safe_name}.pt"
    result_path = output_dir / f"Result_{safe_name}.pt"
    torch.save({
            "parameters": result["equilibrated_parameters"],
            "model_state_dict": result["equilibrated_model_state_dict"],
            "temperature": result["temperature"],
            "run_name": _resolve_run_name(run_name),
        }, equilibrated_path,)
    torch.save({
            "parameters": result["inherent_parameters"],
            "model_state_dict": result["model_state_dict"],
            "temperature": result["temperature"],
            "run_name": _resolve_run_name(run_name),
        }, inherent_path,)
    torch.save(result, result_path)
    return {"equilibrated_checkpoint": str(equilibrated_path), "inherent_checkpoint": str(inherent_path), "result_file": str(result_path), }


def _full_batch_loss(model: torch.nn.Module, task: Any, data: Any, weights: Dict[str, float]) -> torch.Tensor:
    predictions = task.predict(data, model)
    return task.balanced_loss_fn(data, predictions, weights)

""" returns both loss and gradient norm (normally opt.step returns only loss)"""
# TODO: add handling lbfgs and SAM
def _optimizer_step(model, task, data, optimizer, weights):
    optimizer.zero_grad(set_to_none=True)
    loss = _full_batch_loss(model, task, data, weights)
    loss.backward()

    gradient_norm = torch.linalg.vector_norm( torch.cat([ p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]) )

    optimizer.step()
    return float(loss.detach().cpu()), float(gradient_norm.cpu())


def cosine_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return q = <first, second> / (||first|| ||second||)."""

    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if denominator == 0:
        return 0.0
    return float(torch.dot(first, second).div(denominator).cpu())

def get_temperature(cfg, experiment_cfg):
    if experiment_cfg.equilibration_optimizer_name == "langevin":
        return experiment_cfg.equilibration_optimizer_params["temperature"] 
    elif experiment_cfg.equilibration_optimizer_name == "sgd":
        return experiment_cfg.equilibration_optimizer_params["lr"] / cfg.task_params["batch_size"]
    else:
        warnings.warn(f"Temperature is not defined for optimizer " f"'{experiment_cfg.equilibration_optimizer_name}'. Returning None.")
        return None


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
    # some settings
    displacement_every = 100

    if experiment_cfg.equilibration_steps < 0 or experiment_cfg.quench_steps < 0:
        raise ValueError("step counts must be non-negative")
    if experiment_cfg.wandb_log_every < 0:
        raise ValueError("wandb_log_every must be non-negative")
    if experiment_cfg.model_seed is not None:
        set_random_seed(experiment_cfg.model_seed)
    else:
        set_random_seed(experiment_cfg.seed)
    if experiment_cfg.equilibration_optimizer_name in ["lbfgs", "sam"] or experiment_cfg.quench_optimizer_name in ["lbfgs", "sam"]:
        raise NotImplementedError(
            "The LBFGS and SAM optimizers are not currently supported for "
            "inherent-structure experiments."
            "Adjust the _optimizer_step function to support these optimizers if you want to use them."
        )

    device = torch.device(device)
    model_was_provided = model is not None
    model = model or get_model(cfg.model_name, cfg.model_params)
    model = model.to(device)
    if not model_was_provided:
        model.apply(init_weights)
    
    set_random_seed(experiment_cfg.seed) # seed for the dynamics, if model_seed is provided, it will be used for model initialization, otherwise seed will be used for both model and dynamics

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

    record_displacement = "displacement" in experiment_cfg.metrics
    langevin_displacement = []
    quench_displacement = []

    """ First phase: EQUILIBRATE"""
    langevin_start = _parameters_to_vector(model)

    equilibration_params = dict(experiment_cfg.equilibration_optimizer_params or {})
    if experiment_cfg.equilibration_optimizer_name == "langevin":
        equilibration_params["n_examples"] = len(full_data[0])

    optimizer = get_opt(
        experiment_cfg.equilibration_optimizer_name,
        equilibration_params,
        model.parameters(),
    )
    for step in range(experiment_cfg.equilibration_steps):
        if experiment_cfg.equilibration_optimizer_name in ["langevin", "gd"]:
            data = full_data
        else:
            data = task.get_data()
        loss, gradient_norm = _optimizer_step(model, task, data, optimizer, loss_weights)
        if record_displacement and displacement_every > 0 and (step + 1) % displacement_every == 0:
            w = _parameters_to_vector(model)
            delta = torch.mean((w - langevin_start) ** 2).item()
            langevin_displacement.append((step + 1, delta))
        if (experiment_cfg.wandb_log_every > 0 and (step + 1) % experiment_cfg.wandb_log_every == 0):
            _wandb_log( {"langevin/loss": loss, "langevin/gradient_norm": gradient_norm,  "phase_step": step + 1, }, step=step + 1, )

    equilibrated_parameters = _parameters_to_vector(model)
    equilibrated_model_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    """ Second Phase: QUENCH"""
    quench_optimizer = get_opt(
        experiment_cfg.quench_optimizer_name,
        experiment_cfg.quench_optimizer_params, 
        model.parameters(),
    )
    final_loss = float("nan")
    final_gradient_norm = float("nan")
    quench_start = _parameters_to_vector(model)
    for step in range(experiment_cfg.quench_steps):
        if experiment_cfg.quench_optimizer_name in ["langevin", "gd"]:
            data = full_data
        else:
            data = task.get_data()
        final_loss, final_gradient_norm = _optimizer_step(model, task, data, quench_optimizer, loss_weights)

        if record_displacement and displacement_every > 0 and (step + 1) % displacement_every == 0:
            w = _parameters_to_vector(model)
            delta = torch.mean((w - quench_start) ** 2).item()
            quench_displacement.append((step + 1, delta))

        if (experiment_cfg.wandb_log_every > 0 and (step + 1) % experiment_cfg.wandb_log_every == 0):
            _wandb_log(
                {
                    "quench/loss": final_loss,
                    "quench/gradient_norm": final_gradient_norm,
                    "phase_step": step + 1,
                },
                step=experiment_cfg.equilibration_steps + step + 1,
            )
        if (
            experiment_cfg.quench_tolerance > 0
            and final_gradient_norm <= experiment_cfg.quench_tolerance
        ):
            break

    inherent_parameters = _parameters_to_vector(model)
    with torch.no_grad():
        final_loss = float(_full_batch_loss(model, task, full_data, loss_weights).cpu())

    result = {
        "temperature": get_temperature(cfg, experiement_cfg),
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
    if record_displacement:
        result["displacement"] = {
            "langevin": langevin_displacement,
            "quench": quench_displacement,
        }
    if save_dir is not None:
        result["saved_files"] = _save_named_checkpoints(result, save_dir, run_name)
    return result
