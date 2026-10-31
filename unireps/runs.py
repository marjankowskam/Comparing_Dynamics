"""Read-only UniReps run discovery. The Excel 'Unireps' sheet is authoritative."""
from pathlib import Path
import hashlib

import pandas as pd
import torch

from comparing_dynamics.config import Config


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(str((name, tuple(x.shape), x.dtype)).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def torch_load(path, map_location='cpu'):
    # Existing notebooks use this for trusted experiment checkpoints.
    return torch.load(path, map_location=map_location, weights_only=False)


def get_model_state(loaded_model_file):
    """Copied from the working quench notebook."""
    if isinstance(loaded_model_file, torch.nn.Module):
        return loaded_model_file.state_dict()
    if isinstance(loaded_model_file, dict):
        for key in ('model_state_dict', 'state_dict'):
            if key in loaded_model_file:
                return loaded_model_file[key]
        if loaded_model_file and all(torch.is_tensor(v) for v in loaded_model_file.values()):
            return loaded_model_file
    raise TypeError('Could not identify a model state_dict in the loaded Model_*.pt file.')


def safe_filename(text):
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(text))


def load_run_registry(metadata_path):
    df = pd.read_excel(metadata_path, sheet_name='Unireps')
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={'seed)': 'seed'})
    required = {'name', 'batch_size', 'lr', 'task', 'seed', 'weight_decay', 'steps'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing Excel columns: {sorted(missing)}')
    df = df.dropna(subset=['name']).copy()
    df['name'] = df['name'].astype(str).str.strip()
    if df['name'].duplicated().any():
        raise ValueError(f'Duplicate run names in Excel: {df.loc[df.name.duplicated(), "name"].tolist()}')
    return df


def load_config(run_name, checkpoint_dir):
    # Config.load_from_file constructs the literal filename 'Config_' + run_name.
    root = Path(checkpoint_dir)
    for folder in (root, root / 'extended_preparation'):
        if (folder / f'Config_{run_name}').is_file():
            return Config.load_from_file(run_name, data_dir=str(folder) + '/')
    return None


def resolve_parent(run_name, checkpoint_dir, weight_decay=0, use_extended=True):
    root = Path(checkpoint_dir)
    filename = f'Model_{run_name}.pt'
    candidates = []
    if use_extended:
        ext = root / 'extended_preparation'
        candidates = [
            ext / 'fdr_equilibration_v3' / f'{run_name}-progress.pt',
            ext / f'wd_{float(weight_decay or 0):g}' / 'sgd_continuation_v1' / filename,
            ext / filename,
        ]
    candidates.append(root / filename)
    return next((p for p in candidates if p.is_file()), None)


def find_quench(run_name, parent_path, quench_dir, steps=40000, quench_lr=None):
    if parent_path is None:
        return None, 'missing_parent'
    safe_name = safe_filename(run_name)
    # Search only for this Excel-listed run.
    prefix = f'{safe_name}_GD_lr'
    candidates = [
        p for p in Path(quench_dir).glob(f'{prefix}*_steps{steps}*.pt')
        if p.name.startswith(prefix)
    ]
    if quench_lr is not None:
        candidates = [p for p in candidates if p.name.startswith(f'{prefix}{quench_lr:g}_steps{steps}')]
    if not candidates:
        return None, 'missing'

    parent = torch_load(parent_path)
    parent_digest = state_hash(get_model_state(parent))
    del parent
    matches = []
    for path in candidates:
        quench = torch_load(path)
        identity = quench.get('cache_identity') or {}
        if (identity.get('parent_hash') == parent_digest
                and identity.get('max_steps') == steps
                and (quench_lr is None or identity.get('quench_lr') == quench_lr)):
            matches.append(path)
        del quench
    if len(matches) == 1:
        return matches[0], 'matched'
    if len(matches) > 1:
        return None, 'multiple_matches'
    return None, 'unmatched'


def discover_runs(metadata_path, checkpoint_dir, quench_dir, *,
                  run_names=None, task=None, weight_decay=None,
                  use_extended=True, quench_steps=40000, quench_lr=None,
                  verify_quenches=False):
    """Fast registry-based lookup. Only load checkpoint tensors if explicitly verifying quenches.

    Filter to the requested Excel runs BEFORE accessing Drive files.
    """
    registry = load_run_registry(metadata_path)
    if run_names is not None:
        requested = {str(n) for n in run_names}
        unknown = requested - set(registry['name'])
        if unknown:
            raise ValueError(f'Runs absent from Excel: {sorted(unknown)}')
        registry = registry[registry['name'].isin(requested)]
    if task is not None:
        registry = registry[registry['task'].astype(str).str.lower() == str(task).lower()]
    if weight_decay is not None:
        wd_col = pd.to_numeric(registry['weight_decay'], errors='coerce').fillna(0.0)
        registry = registry[(wd_col - float(weight_decay)).abs() < 1e-12]
    rows = []
    for row in registry.itertuples(index=False):
        run_name = row.name
        cfg = load_config(run_name, checkpoint_dir)
        if cfg is not None and str(cfg.opt_name).lower() != 'sgd':
            continue
        wd = float(row.weight_decay) if pd.notna(row.weight_decay) else 0.0
        parent = resolve_parent(run_name, checkpoint_dir, weight_decay=wd, use_extended=use_extended)
        if verify_quenches:
            path, status = find_quench(run_name, parent, quench_dir, steps=quench_steps, quench_lr=quench_lr)
        else:
            # Do not open large parent/quench .pt files during discovery.
            path, status = None, 'not_checked' if parent else 'missing_parent'
        rows.append({
            'run_id': run_name,
            'task_name': row.task,
            'batch_size': row.batch_size,
            'lr': row.lr,
            'seed': row.seed,
            'weight_decay': wd,
            'steps': row.steps,
            'config_found': cfg is not None,
            'parent_path': str(parent) if parent else None,
            'preparation': ('extended' if 'extended_preparation' in parent.parts else 'original') if parent else None,
            'quench_path': str(path) if path else None,
            'quench_status': status,
        })
    return pd.DataFrame(rows)
