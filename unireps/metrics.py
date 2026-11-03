"""UniReps metrics. Numerical definitions extracted from the working CKA notebooks.

Original sources: main_parent_parent_child_child_network_CKA and
fashionmnist_run_comparison_fixed. The cache reads existing parent_noise_D.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from scipy.stats import t as student_t
from .runs import get_model_state, torch_load, state_hash
from .quench import build_model_from_state, build_task_for_model, load_run_config

ACTIVATION_TYPES = (torch.nn.ReLU, torch.nn.Tanh, torch.nn.GELU, torch.nn.Sigmoid, torch.nn.SiLU)


def center_features(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(x.shape[0], -1)
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x, y, eps=1e-12):
    x, y = center_features(x), center_features(y)
    xy, xx, yy = x.T @ y, x.T @ x, y.T @ y
    denominator = np.linalg.norm(xx, 'fro') * np.linalg.norm(yy, 'fro')
    return float(np.sum(xy * xy) / denominator) if denominator > eps else np.nan


def offdiag_mean(C):
    C = np.asarray(C, dtype=float)
    if len(C) < 2:
        return np.nan
    return float(C[~np.eye(len(C), dtype=bool)].mean())


def jackknife_offdiag_ci(C):
    C = np.asarray(C, dtype=float)
    n = len(C)
    value = offdiag_mean(C)

    if n < 3 or not np.isfinite(value):
        return value, np.nan, np.nan

    deleted = []
    for i in range(n):
        keep = np.arange(n) != i
        deleted.append(offdiag_mean(C[np.ix_(keep, keep)]))

    deleted = np.asarray(deleted)
    se = np.sqrt((n - 1) / n * np.sum(
        (deleted - deleted.mean()) ** 2
    ))
    radius = student_t.ppf(0.975, n - 1) * se

    return value, value - radius, value + radius


def geometric_mean(values):
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError('T_eff must be positive.')
    return float(np.exp(np.log(values).mean()))


def state_at(states, step):
    value = states.get(step, states.get(str(step)))
    if value is not None:
        # Both saved formats occur in the comparison notebook.
        return value.get('metrics', value)
    raise KeyError(f'Quench file has no state at step {step}.')


@torch.no_grad()
def extract_representations(model, task, data):
    outputs, handles = {}, []

    def make_hook(name):
        def hook(_module, _inputs, output):
            h = output.detach()
            if h.ndim > 2:
                h = h.flatten(start_dim=1)
            outputs[name] = h.cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, ACTIVATION_TYPES):
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        _ = task.predict(data, model)
    finally:
        for handle in handles:
            handle.remove()

    return outputs


def layer_mean_cka(rep_a, rep_b, layers):
    values = np.asarray([
        linear_cka(rep_a[layer], rep_b[layer])
        for layer in layers
    ], dtype=float)

    if not np.isfinite(values).all():
        return np.nan
    return float(values.mean())


def noise_D(model, task, full_data, chunk=200):
    """Same full-data, per-parameter D definition as the FashionMNIST code."""
    model.eval()
    device = next(model.parameters()).device
    x, y = full_data[:2]
    params = {k: v.detach() for k, v in model.named_parameters()}
    M = sum(p.numel() for p in params.values())

    x0, y0 = x[:8].to(device), y[:8].to(device)
    with torch.no_grad():
        if not torch.allclose(
            model(x0), task.predict((x0, y0), model),
            atol=1e-5, rtol=1e-5):
            raise RuntimeError('model(x) and task.predict use different preprocessing.')

    def loss_one(p, xs, ys):
        logits = functional_call(model, p, (xs.unsqueeze(0),))
        return F.cross_entropy(logits, ys.unsqueeze(0))

    per_example_grad = vmap(grad(loss_one), in_dims=(None, 0, 0))
    grad_sum = {
        k: torch.zeros_like(v, dtype=torch.float64)
        for k, v in params.items()
    }
    sum_sq = 0.0

    for start in range(0, len(x), chunk):
        gradients = per_example_grad(
            params,
            x[start:start + chunk].to(device),
            y[start:start + chunk].to(device),
        )
        for key, value in gradients.items():
            g = value.double()
            sum_sq += g.square().sum().item()
            grad_sum[key] += g.sum(dim=0)
        del gradients, g

    N = len(x)
    full_grad_sq = sum(
        (value / N).square().sum().item()
        for value in grad_sum.values()
    )
    D = (sum_sq / N - full_grad_sq) / M
    if not np.isfinite(D) or D <= 0:
        raise ValueError(f'Invalid measured D: {D}')
    return float(D)


def baseline_correct_cka(value, layer, cka0, clip=False):
    base = cka0[layer]
    corrected = (value - base) / (1.0 - base)
    return float(np.clip(corrected, 0.0, 1.0)) if clip else float(corrected)


def jackknife_matrix_ci(C, statistic=offdiag_mean):
    C = np.asarray(C, dtype=float)
    n = len(C)
    value = statistic(C)
    if n < 3 or not np.isfinite(value):
        return value, np.nan, np.nan

    deleted = []
    for i in range(n):
        keep = np.arange(n) != i
        deleted.append(statistic(C[np.ix_(keep, keep)]))
    deleted = np.asarray(deleted, dtype=float)

    se = np.sqrt((n - 1) / n * np.sum((deleted - deleted.mean()) ** 2))
    radius = student_t.ppf(0.975, n - 1) * se
    return value, value - radius, value + radius



def weight_metrics(parent_model, child_model):
    """Same q_w and relative displacement definitions as main notebook cell 30."""
    wp = torch.nn.utils.parameters_to_vector(parent_model.parameters()).detach().cpu().double()
    wc = torch.nn.utils.parameters_to_vector(child_model.parameters()).detach().cpu().double()
    q_w = float((wp @ wc) / (wp.norm() * wc.norm()))
    relative_displacement = float((wc - wp).norm() / wp.norm())
    return q_w, relative_displacement


def effective_temperatures(runs, checkpoint_dir, full_data, cache_path, device, chunk=1000):
    """Exact FashionMNIST D cache: reuse only the same run UID and parent hash.

    Runs must already have been filtered by the Excel registry (runs.py).
    Existing parent_noise_D.csv remains usable, and new entries are appended.
    """
    cache_path = Path(cache_path)
    columns = ['uid', 'run_id', 'experiment', 'parent_hash', 'D', 'T_eff']
    d_cache = pd.read_csv(cache_path) if cache_path.is_file() else pd.DataFrame(columns=columns)
    cache = {(r.uid, r.parent_hash): float(r.D) for r in d_cache.itertuples(index=False)
             if pd.notna(r.D)}
    rows = []
    for row in runs.itertuples(index=False):
        uid = str(getattr(row, 'uid', f'wd{float(row.weight_decay or 0):g}::{row.run_id}' if 'fashion' in str(row.task_name).lower() else row.run_id))
        parent_state = None
        parent_hash = getattr(row, 'parent_hash', None)
        if not isinstance(parent_hash, str) or not parent_hash:
            parent_state = get_model_state(torch_load(row.parent_path))
            parent_hash = state_hash(parent_state)
        key = (uid, parent_hash)
        if key in cache:
            D = cache[key]
        else:
            if parent_state is None:
                parent_state = get_model_state(torch_load(row.parent_path))
            cfg = load_run_config(row.run_id, checkpoint_dir)
            model = build_model_from_state(parent_state, cfg, device)
            task = build_task_for_model(model, cfg, device)
            D = noise_D(model, task, full_data, chunk=chunk)
            del task, model, parent_state
        T_eff = (float(row.lr) / float(row.batch_size)) * D
        if not np.isfinite(T_eff) or T_eff <= 0:
            raise ValueError(f'{row.run_id}: invalid effective temperature {T_eff}')
        rows.append(dict(uid=uid, run_id=row.run_id,
                         experiment=getattr(row, 'experiment', None),
                         parent_hash=parent_hash, D=D, T_eff=T_eff))
        if key not in cache:
            progress = pd.concat([d_cache, pd.DataFrame(rows)], ignore_index=True)
            progress = progress.drop_duplicates(['uid', 'parent_hash'], keep='last')
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            progress.to_csv(cache_path, index=False)
    return pd.DataFrame(rows)


def own_parent_cka(reps, runs, layers, cka0=None):
    """Same per-layer own-parent CKA and optional random-baseline correction."""
    rows = []
    for row in runs.itertuples(index=False):
        uid = str(getattr(row, 'uid', row.run_id))
        for layer in layers:
            value = linear_cka(reps[uid]['parent'][layer], reps[uid]['child'][layer])
            entry = dict(uid=uid, run_id=row.run_id, layer=layer, cka=value)
            if cka0 is not None:
                entry['cka_baseline_corrected'] = baseline_correct_cka(value, layer, cka0)
                entry['cka_baseline_corrected_clipped'] = baseline_correct_cka(value, layer, cka0, clip=True)
            rows.append(entry)
    return pd.DataFrame(rows)


def pairwise_cka(runs, reps, layers):
    """Parent–parent and child–child CKA, grouped by preparation LR and batch."""
    rows = []
    group_cols = [c for c in ('task_name', 'weight_decay', 'lr', 'batch_size') if c in runs]
    for keys, group in runs.groupby(group_cols, sort=False):
        group = group.sort_values('seed')
        ids = [str(x) for x in group.get('uid', group['run_id'])]
        for layer in layers:
            for kind in ('parent', 'child'):
                C = np.array([[linear_cka(reps[a][kind][layer], reps[b][kind][layer])
                               for b in ids] for a in ids])
                mean, low, high = jackknife_matrix_ci(C)
                rows.append(dict(**dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                                 layer=layer, kind=kind, n_seeds=len(ids),
                                 mean=mean, low=low, high=high,
                                 T_eff=geometric_mean(group.T_eff) if 'T_eff' in group else np.nan))
    return pd.DataFrame(rows)
