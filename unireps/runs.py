# Automatically discover parent checkpoints, extended parents, configurations and matching quenches. 

# Example notebook usage
# from unireps.runs import discover_runs

# runs = discover_runs(
#     metadata_path=REPO_ROOT / 'data' / 'Comparing_Dynamics (1).xlsx',
#     checkpoint_dir=CHECKPOINT_DIR,
#     quench_dir=QUENCH_DIR
# )
# display(runs)

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


def get_model_state(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return value
            if isinstance(value, torch.nn.Module):
                return value.state_dict()
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError('Checkpoint has no recognised model state dictionary')


def load_run_registry(metadata_path):
    df = pd.read_excel(metadata_path, sheet_name='Unireps')
    df.columns = df.columns.str.strip()
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
    safe_name = ''.join(c if c.isalnum() or c in '._-' else '_' for c in run_name)
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

    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    parent_digest = state_hash(get_model_state(parent))
    del parent
    matches = []
    for path in candidates:
        quench = torch.load(path, map_location='cpu', weights_only=False)
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


def discover_runs(metadata_path, checkpoint_dir, quench_dir, *, use_extended=True, quench_steps=40000, quench_lr=None):
    rows = []
    for row in load_run_registry(metadata_path).itertuples(index=False):
        run_name = row.name
        cfg = load_config(run_name, checkpoint_dir)
        wd = float(row.weight_decay) if pd.notna(row.weight_decay) else 0.0
        parent = resolve_parent(run_name, checkpoint_dir, weight_decay=wd, use_extended=use_extended)
        path, status = find_quench(run_name, parent, quench_dir, steps=quench_steps, quench_lr=quench_lr)
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
# from pathlib import Path
# import pandas as pd
# from comparing_dynamics.config import Config

# def load_run_registry(metadata_path):
#     df = pd.read_excel(metadata_path, sheet_name='Unireps')
#     df.columns = df.columns.str.strip()
#     if 'name' not in df: raise KeyError("Missing 'name' column in Unireps sheet")
#     return df.dropna(subset=['name']).drop_duplicates('name').rename(columns={'name': 'run_id'})


# def discover_checkpoints(checkpoint_dir):
#     root = Path(checkpoint_dir)
#     extended = root / 'extended_preparation'
#     files = list(root.glob('Model_*.pt'))
#     if extended.exists():
#         files += list(extended.rglob('Model_*.pt'))
#         files += list(extended.rglob('*-progress.pt'))

#     found = {}
#     for path in files:
#         name = path.name
#         run_id = name.removeprefix('Model_').removesuffix('.pt') if name.startswith('Model_') else name.removesuffix('-progress.pt')
#         found.setdefault(run_id, []).append(path)
#     return found


# def load_config(run_id, checkpoint_dir):
#     root = Path(checkpoint_dir)
#     for folder in [root, root / 'extended_preparation']:
#         try:
#             return Config.load_from_file(run_id, str(folder) + '/')
#         except FileNotFoundError:
#             continue
#     return None


# def resolve_parent(run_id, cfg, checkpoint_dir):
#     root = Path(checkpoint_dir)
#     ext = root / 'extended_preparation'
#     wd = (cfg.task_params.get('weight_decay') or 0) if cfg else 0
#     filename = f'Model_{run_id}.pt'
#     candidates = [
#         ext / 'fdr_equilibration_v3' / f'{run_id}-progress.pt',
#         ext / f'wd_{wd:g}' / 'sgd_continuation_v1' / filename,
#         ext / filename,
#         root / filename,
#     ]
#     for path in candidates:
#         if path.is_file(): return path
#     return None


# import hashlib
# import torch

# def state_hash(state):
#     h = hashlib.sha256()
#     for name, value in sorted(state.items()):
#         x = value.detach().cpu().contiguous()
#         h.update(str((name, tuple(x.shape), x.dtype)).encode())
#         h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
#     return h.hexdigest()

# def get_model_state(checkpoint):
#     if isinstance(checkpoint, torch.nn.Module):
#         return checkpoint.state_dict()
#     if isinstance(checkpoint, dict):
#         for key in ('model_state_dict', 'state_dict'):
#             if key in checkpoint: return checkpoint[key]
#         if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
#             return checkpoint
#     raise TypeError('Unrecognised model checkpoint')


# def find_quench(run_id, parent_path, quench_dir, steps=40000, lr=None):
#     safe_name = ''.join( c if c.isalnum() or c in '._-' else '_' for c in str(run_id))
#     candidates = sorted( Path(quench_dir).glob(f'{safe_name}_GD_lr*_steps{steps}_*.pt'))

#     if not candidates:
#         return None, 'missing'

#     parent_state = get_model_state( torch.load(parent_path, map_location='cpu', weights_only=False) )
#     parent_hash = state_hash(parent_state)

#     matches = []
#     for path in candidates:
#         q = torch.load(path, map_location='cpu', weights_only=False)
#         identity = q.get('cache_identity') or {}

#         if identity.get('parent_hash') != parent_hash:
#             continue
#         if identity.get('max_steps') != steps:
#             continue
#         if lr is not None and identity.get('quench_lr') != lr:
#             continue

#         matches.append(path)

#     if len(matches) == 1:
#         return matches[0], 'matched'
#     if len(matches) > 1:
#         return None, 'multiple_matches'
#     return None, 'stale'



# def discover_runs(metadata_path, checkpoint_dir, quench_dir, steps=40000):
#     registry = load_run_registry(metadata_path)
#     found = discover_checkpoints(checkpoint_dir)
#     metadata = registry.set_index('run_id').to_dict('index')
#     names = sorted(set(registry.run_id) | set(found))
#     rows = []

#     for name in names:
#         cfg = load_config(name, checkpoint_dir)
#         parent = resolve_parent(name, cfg, checkpoint_dir)
#         if parent is None: continue

#         info = metadata.get(name, {})
#         task = cfg.task_name if cfg else info.get('task')
#         lr = cfg.opt_params.get('lr') if cfg else info.get('lr')
#         batch = cfg.task_params.get('batch_size') if cfg else info.get('batch_size')
#         wd = cfg.task_params.get('weight_decay') if cfg else info.get('weight_decay')
#         seed = cfg.initial_seed if cfg else info.get('seed')
#         quench_path, quench_status = find_quench( name, parent, quench_dir, steps)

#         rows.append(dict(
#             run_id=name, task_name=task, lr=lr, batch_size=batch,
#             weight_decay=wd, seed=seed,
#             parent_path=str(parent), parent_source='extended' if 'extended_preparation' in parent.parts else 'original',
#             config_found=cfg is not None, registry_source='excel' if name in metadata else 'checkpoint_only',
#             quench_path=str(quench_path) if quench_path else None,
#             quench_status=quench_status,
#         ))

#     return pd.DataFrame(rows)
