# Comparing_Dynamics

`config.py`: specifies the parameters on which the experiement is run. Should uniquely identify the specifications of the run. Specifies the **task, optimizer, model, and experiment** names and their respective parameters.

`\tasks\`: specify the problem eg. mnist, fashionmnis, pinn, etc. Each class has functionality for loading the dataset, evaluating the loss function and evaluate the model.

`models.py`: has architectures eg. MLP, CNN

`train_utilis.py`: are helper functions for training and handling loading the configuration

`experiement.py` is part of the `experiment` and its the functionality that specifies which metrics are collected and at what points in training.

`train`/`run_example` perform the training based on supplied experiment configuration `config`. They save as files the config, all collected metrics, final weights and optimizer status. The name run is often random - supplied by the wandb.


