# Comparing Dynamics

This repository contains experiments for comparing neural-network learning dynamics.

## Layout

```text
comparing_dynamics/
├── config.py              # serializable run configuration
├── models.py              # model architectures
├── optimizer.py           # optimizer construction
├── train_utils.py         # shared task and training helpers
├── tasks/                 # datasets and task-specific loss/evaluation code
└── experiments/
	├── base.py            # multi-trajectory experiment interfaces
	└── ...                # individual studies
scripts/
├── run_experiment.py      # command-line entry point
└── legacy_train.py        # original training script, retained for migration
notebooks/
└── inherent_structures.ipynb # primary Colab/notebook workflow
```

The package layout means experiment modules can import shared code without depending on the current working directory:

```python
from comparing_dynamics.models import get_model
from comparing_dynamics.tasks.mnist import MNISTTask
from comparing_dynamics.experiments.base import Experiment, MetricsConfig
```

Install the repository in editable mode from its root:

```text
pip install -e .
```

The recommended way to run the inherent-structure study is the notebook [notebooks/inherent_structures.ipynb](notebooks/inherent_structures.ipynb). It runs the temperature sweep, creates the W&B runs, saves the named checkpoints, and plots the summary directly from Python cells.

The command-line wrapper is optional and can be run as a module:

```text
python -m scripts.run_experiment --config path/to/config.json
```

## Experiment and trajectory logging

`comparing_dynamics.experiments.base` provides `Experiment`, `TrajectoryResult`, and `MetricsConfig`. Use `MetricsConfig(save_every_step=False)` for final metrics only. Use `metric_names` to retain only selected per-step metrics while still returning final metrics from each trajectory.

Each experiment should own its study-specific parameter sweep and call shared model, task, optimizer, and training APIs. Keep generated data, checkpoints, and plots outside the source package, for example under `runs/<experiment-name>/<trajectory-id>/`.

## Inherent-structure experiment

The inherent-structure experiment in `comparing_dynamics.experiments.inherent_structures` runs one trajectory in two phases:

1. Start from a freshly initialized network and run full-batch Langevin dynamics at temperature `T`.
2. Snapshot the equilibrated parameters, remove the noise, and run full-batch SGD until the configured quench budget or gradient tolerance is reached.

The result contains the final loss, final gradient norm, the cosine overlap `overlap_q` between the equilibrated state and the inherent structure, and parameter snapshots for all three states. The command-line wrapper is:

```text
python -m scripts.inherent_structures_train --config Config_run.json --temperature 0.1 --equilibration-steps 1000 --quench-steps 5000 --langevin-lr 0.001 --quench-lr 0.01 --run-name temperature-0.1 --save-dir runs/inherent-structures
```

The run name defaults to the active W&B run name when `wandb.init()` is active. Each run saves `Equilibrated_<run-name>.pt`, `Inherent_<run-name>.pt`, and `Result_<run-name>.pt`. The first two files contain both flattened parameters and a reloadable model state dictionary.

For a temperature sweep, call `run_inherent_structure` once per temperature and assign a distinct seed or trajectory identifier in the surrounding experiment module. The full-batch data is deliberate here: the quench is meant to define an inherent structure for the same loss landscape sampled during equilibration. Both phases use `comparing_dynamics.optimizer.get_opt`; the Langevin phase currently permits `sgd` only because its isotropic noise covariance is derived for SGD. The quench phase can use any optimizer supported by the catalog.


