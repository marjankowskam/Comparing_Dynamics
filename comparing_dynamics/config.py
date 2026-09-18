import json

class Config:
    def __init__(
        self,
        initial_seed=19,

        task_name = "",
        task_params = None,

        opt_name="",
        opt_params=None,

        balancer_type="Fixed",
        update_balancer_every_iter=100,

        model_name = "",
        model_params = None,

        experiment_name = "",
        experiment_params = None,

        num_epochs=100,
    ):
        self.initial_seed = initial_seed
        
        self.task_name = task_name
        self.task_params = task_params or {}

        self.opt_name = opt_name
        self.opt_params = opt_params or {}

        self.balancer_type = balancer_type
        self.update_balancer_every_iter = update_balancer_every_iter

        self.model_name = model_name
        self.model_params = model_params or {}

        self.experiment_name = experiment_name
        self.experiment_params = experiment_params or {}

        self.num_epochs = num_epochs

    def save_to_file(self, filename: str, data_dir=''):
        with open(data_dir+'Config_'+filename, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load_from_file(cls, filename: str, data_dir=''):
        with open(data_dir+'Config_'+filename, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self):
        return f"{self.__dict__}"
    



# """Example: Regression"""
# cfg = Config(

#     task_name = "pde_regression",
#     task_params = {
#         "pde_name": 'convection',
#         "pde_params": ["beta", "40"],
#         "num_x": 257,
#         "num_t": 101,
#         "num_res": 10000,
#         "loss_name": "mse"
#     },

#     model_name = "pinn",
#     model_params = {
#         "hidden_dim": 200,
#         "num_layers": 4,
#     },

#     opt_name = "adam",
#     opt_params={
#         "lr": 0.001
#     },

#     experiment_name = "displacement",
#     experiment_params = {
#         "hessian_n_top": 3,
#         "hessian_n_bottom": 0,
#         "hessian_checkpoints": [],
#         "hessian_eigv_tol": 1e-3,
#     },

#     num_epochs = 10*1000,
# )

# """Example: PINN"""
# cfg = Config(

#     task_name = "pinn",
#     task_params = {
#         "pde_name": 'convection',
#         "pde_params": ["beta", "40"],
#     },

#     model_name = "pinn",
#     model_params = {
#         "hidden_dim": 200,
#         "num_layers": 4,
#     },

#     opt_name = "adam",
#     opt_params={
#         "lr": 0.001
#     },

#     experiment_name = "displacement",
#     experiment_params = {
#         "hessian_n_top": 3,
#         "hessian_n_bottom": 0,
#         "hessian_checkpoints": [],
#         "hessian_eigv_tol": 1e-3,
#     },

#     num_epochs = 3,
# )

# """Example: MNIST"""
# Config(
#     task_name = "mnist",
#     task_params = {
#         "batch_size": 128, #if None, full batch used.
#     },
#     model_name = "mnistMLP",
#     model_params = {
#         "hidden_dim": [100, 100, 10],
#         "activation_fn": nn.ReLU,
#     },
#     opt_name="sgd",
#     opt_params={
#         "lr": 0.01
#     },
#     experiment_name = "displacement",
#     experiment_params = {
#         "hessian_n_top": 3,
#         "hessian_n_bottom": 0,
#         "hessian_checkpoints": [],
#         "hessian_eigv_tol": 1e-3,
#     },
#     num_epochs = 10,
# )
