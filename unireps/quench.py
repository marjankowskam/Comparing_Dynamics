"""UniReps full-batch GD quench, extracted from the 6thOct quench/CKA notebook.

Uses the existing Comparing_Dynamics model/task implementations and original
quench cache identity. Does not discover new runs: input rows come from Excel.
"""

import hashlib
import json
import copy
from pathlib import Path

import torch
from tqdm.auto import tqdm

from comparing_dynamics.models import get_model
from comparing_dynamics.train_utils import get_task
from comparing_dynamics.config import Config
from .runs import get_model_state, safe_filename, state_hash, torch_load

DEFAULT_RECORD_STEPS = (0, 1_000, 5_000, 10_000, 20_000, 40_000)


def stable_signature(value):
    return json.dumps(value, sort_keys=True, default=str)


def canonical_result(q):
    """Read both old mock-trial and main notebook state formats."""
    q = dict(q)
    q['states'] = {
        int(step): ({**v['metrics'], 'representations': v['reps']}
                    if 'reps' in v else v)
        for step, v in q['states'].items()
    }
    return q


def load_run_config(run_id, checkpoint_dir):
    """Load the original saved config (not a guessed reconstruction)."""
    return Config.load_from_file(run_id, str(Path(checkpoint_dir)) + '/')


def build_model_from_state(state_dict, run_cfg, device):
    model = get_model(run_cfg.model_name, run_cfg.model_params).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


def build_task_for_model(model, run_cfg, device):
    task = get_task(run_cfg.task_name, run_cfg.task_params, device)
    task.initialize([name for name, _ in model.named_parameters()], model)
    return task


def parameters_to_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def full_batch_loss(model, task, data, weights):
    predictions = task.predict(data, model)
    return task.balanced_loss_fn(data, predictions, weights)


def loss_and_grad_norm(model, task, data, weights):
    model.zero_grad(set_to_none=True)
    loss = full_batch_loss(model, task, data, weights)
    loss.backward()
    grads = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.cat(grads)) if grads else loss.new_tensor(0.)
    return float(loss.detach().cpu()), float(norm.detach().cpu())


@torch.no_grad()
def test_accuracy(model, task, device):
    return task.evaluate(model, device)['accuracy']


@torch.no_grad()
def maybe_accuracy(model, task, data):
    try:
        predictions = task.predict(data, model)
        labels = data[1] if isinstance(data, (tuple, list)) and len(data) >= 2 else None
        if (torch.is_tensor(predictions) and predictions.ndim == 2 and
                torch.is_tensor(labels) and labels.ndim == 1 and
                predictions.shape[0] == labels.shape[0]):
            return float((predictions.argmax(dim=1) == labels).float().mean().cpu())
    except Exception:
        pass
    return float('nan')


@torch.no_grad()
def extract_representations_from_model(model, task, eval_data):
    """Same activation hooks and logits as the original notebook."""
    outputs, handles = {}, []

    def make_hook(name):
        def hook(_module, _inputs, output):
            h = output.detach()
            if h.ndim > 2:
                h = h.flatten(start_dim=1)
            outputs[name] = h.cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.Tanh, torch.nn.GELU,
                               torch.nn.Sigmoid, torch.nn.SiLU)):
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        model.eval()
        outputs['logits'] = task.predict(eval_data, model).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.train()  # Original notebook restores train mode after recording.
    return outputs


def record_state(model, task, full_data, weights, parent_vector, step, eval_data, device):
    loss, grad_norm = loss_and_grad_norm(model, task, full_data, weights)
    w = parameters_to_vector(model)
    return {
        'step': step,
        'loss': loss,
        'grad_norm': grad_norm,
        'weight_norm': float(torch.linalg.vector_norm(w).cpu()),
        'normalized_displacement': float(
            (torch.linalg.vector_norm(w - parent_vector) /
             torch.linalg.vector_norm(parent_vector)).cpu()),
        'test_accuracy': test_accuracy(model, task, device),
        'train_accuracy': maybe_accuracy(model, task, full_data),
        'representations': extract_representations_from_model(model, task, eval_data),
    }


def quench_state_dict_gd(parent_state_dict, run_cfg, eval_data, *, device,
                          lr, max_steps=40_000, record_steps=DEFAULT_RECORD_STEPS,
                          progress=True):
    """The original full-batch GD quench; optimizer is plain SGD without momentum."""
    record_steps = set(record_steps) | {0, max_steps}
    model = build_model_from_state(parent_state_dict, run_cfg, device)
    model.train()
    task = build_task_for_model(model, run_cfg, device)
    full_data = task.get_full_data()
    weights = task.initial_loss_weights()
    parent_vector = parameters_to_vector(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    states = {0: record_state(model, task, full_data, weights, parent_vector, 0, eval_data, device)}
    for step in tqdm(range(1, max_steps + 1), desc='Full-batch GD', leave=False,
                     disable=not progress):
        optimizer.zero_grad(set_to_none=True)
        loss = full_batch_loss(model, task, full_data, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite quench loss at step {step}')
        loss.backward()
        optimizer.step()
        if step in record_steps:
            states[step] = record_state(model, task, full_data, weights, parent_vector,
                                        step, eval_data, device)
    final = states[max_steps]
    return {
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'parameters': parameters_to_vector(model).cpu(),
        'parent_parameters': parent_vector.cpu(),
        'states': states,
        'parent_loss': states[0]['loss'],
        'parent_grad_norm': states[0]['grad_norm'],
        'parent_accuracy': states[0]['test_accuracy'],
        'final_loss': final['loss'],
        'final_grad_norm': final['grad_norm'],
        'final_accuracy': final['test_accuracy'],
        'steps': max_steps,
        'quench_lr': lr,
    }


def load_or_run_quench(row, *, checkpoint_dir, quench_dir, eval_data, eval_indices,
                       eval_seed, device, quench_lr=None, max_steps=40_000,
                       record_steps=DEFAULT_RECORD_STEPS, analysis_dir=None,
                       allow_legacy=True, run_missing=True, progress=True):
    """Original tagged-cache semantics, with optional verified legacy reuse.

    The caller must supply an Excel-selected run row, the shared evaluation
    examples/indices, and the desired dataset-specific quench learning rate.
    Existing tagged caches are loaded without running GD. Nothing is silently
    replaced; a mismatched identity is an error.
    """
    if quench_lr is None:
        quench_lr = {'mnist': 0.01, 'fashion_mnist': 0.003}[str(row.task_name)]
    quench_dir = Path(quench_dir)
    parent_state = get_model_state(torch_load(row.parent_path))
    cfg = load_run_config(row.run_id, checkpoint_dir)
    if str(cfg.task_name).lower().replace('-', '_') in ('underparametrized_mnist', 'mnist_underparametrized'):
        cfg = copy.deepcopy(cfg)
        cfg.task_name = 'mnist'  # Architecture is already specified by cfg.model_params.
    identity = dict(
        parent_hash=state_hash(parent_state),
        config=stable_signature(vars(cfg)),
        eval_seed=eval_seed,
        eval_indices=torch.as_tensor(eval_indices).cpu().tolist(),
        quench_lr=quench_lr,
        max_steps=max_steps,
    )
    tag = hashlib.sha256(stable_signature(identity).encode()).hexdigest()[:16]
    stem = f'{safe_filename(row.run_id)}_GD_lr{quench_lr:g}_steps{max_steps}'
    path = quench_dir / f'{stem}_{tag}.pt'
    if path.is_file():
        q = canonical_result(torch_load(path))
        if q.get('cache_identity') != identity:
            raise ValueError(f'Cache identity mismatch: {path}')
        return q, path, 'cached'

    if allow_legacy:
        # Original InfTemp quench files can live next to the parent models.
        legacy = [quench_dir / f'{stem}.pt', Path(checkpoint_dir) / f'{stem}.pt']
        if analysis_dir is not None:
            legacy.append(Path(analysis_dir) / 'relaxation_mock_trial' /
                          f'{safe_filename(row.run_id)}_lr{quench_lr:g}_steps{max_steps}.pt')
        for old_path in dict.fromkeys(legacy):
            if not old_path.is_file():
                continue
            q = canonical_result(torch_load(old_path))
            if {0, max_steps} - set(q['states']):
                continue
            if int(q.get('steps', max_steps)) != max_steps or float(q.get('quench_lr', quench_lr)) != quench_lr:
                continue
            initial = q['states'][0]
            # Verify the *actual initial weights* for InfTemp's saved format.
            if 'model_state_dict' in initial:
                matches = state_hash(get_model_state(initial)) == identity['parent_hash']
            elif (q.get('cache_identity') or {}).get('parent_hash') is not None:
                matches = q['cache_identity']['parent_hash'] == identity['parent_hash']
            elif q.get('parent_hash') is not None:
                matches = q['parent_hash'] == identity['parent_hash']
            elif 'parent_parameters' in q:
                model = build_model_from_state(parent_state, cfg, device)
                matches = torch.equal(parameters_to_vector(model).cpu(), q['parent_parameters'].cpu())
                del model
            elif 'representations' in initial:
                model = build_model_from_state(parent_state, cfg, device)
                task = build_task_for_model(model, cfg, device)
                current = extract_representations_from_model(model, task, eval_data)
                del model, task
                saved = initial['representations']
                matches = current.keys() == saved.keys() and all(
                    current[k].shape == saved[k].shape and
                    torch.allclose(current[k], saved[k], rtol=1e-5, atol=1e-6)
                    for k in current)
            else:
                matches = False  # Never trust a filename alone.
            if not matches:
                continue
            # Normalise the returned object in memory; don't modify old checkpoints.
            if 'model_state_dict' not in q and 'model_state_dict' in q['states'][max_steps]:
                q['model_state_dict'] = q['states'][max_steps]['model_state_dict']
            return q, old_path, 'reused legacy'

    if not run_missing:
        raise FileNotFoundError(f'No quench matches this exact parent and evaluation setup: {stem}')
    q = quench_state_dict_gd(parent_state, cfg, eval_data, device=device, lr=quench_lr,
                            max_steps=max_steps, record_steps=record_steps, progress=progress)
    q.update(run_id=row.run_id, parent_path=str(row.parent_path), cache_identity=identity)
    if hasattr(row, 'temperature'):
        q['temperature'] = float(row.temperature)
    else:
        q['temperature'] = float('inf') if str(row.run_id).startswith('InfTemp_') else float(row.lr) / float(row.batch_size)
    quench_dir.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix('.tmp')
    torch.save(q, temp_path)
    temp_path.replace(path)
    return q, path, 'new'
