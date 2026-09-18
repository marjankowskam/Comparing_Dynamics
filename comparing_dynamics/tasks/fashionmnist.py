from .base import Task
import torch
import torch.nn.functional as F
import random
from torch.func import functional_call
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

class FashionMNISTTask(Task):
    def __init__(self, batch_size, device, data_dir="/data/", random_baseline=False):
        """
        batch_size: size for mini batching (if batch_size is None - full dataset learning)
        random_baseline: shuffles pixels and labels to test for impact of data structure on hessian structure
        """
        super().__init__(device)

        self.batch_size = batch_size

        # Dataset (FashionMNIST normalization values)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        self.full_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )

        self.test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )

        if random_baseline:
            print(f"Using randomized version of FashionMNIST: shuffled pixels and labels")

            # Random labels
            self.full_dataset.targets = torch.randint(0, 10, self.full_dataset.targets.shape)

            # Pixel permutation (same permutation for all images) - preserves scale
            perm = torch.randperm(28 * 28)

            data = self.full_dataset.data.view(-1, 28 * 28)
            data = data[:, perm]
            self.full_dataset.data = data.view(-1, 28, 28)


        # Full fixed batch for Hessian
        # self.fixed_loader = DataLoader(self.full_dataset, batch_size=10000, shuffle=True, drop_last=True)
        # self.fixed_images, self.fixed_targets = next(iter(self.fixed_loader))
        # self.fixed_images = self.fixed_images.to(device)
        # self.fixed_targets = self.fixed_targets.to(device)

        #  Train test (if batch_size is None - use the Hessian one)
        if batch_size is not None:
            self.train_loader = DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.train_data_gen = self._infinite_train_generator()

        # Test set
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
        self.test_images, self.test_targets = next(iter(self.test_loader))
        self.test_images = self.test_images.to(device)
        self.test_targets = self.test_targets.to(device, dtype=torch.long)

        # For stateless calls
        self.param_names = None
        self.model = None

    # Infinite generator for mini-batches
    def _infinite_train_generator(self):
        while True:
            for xb, yb in self.train_loader:
                yield xb.to(self.device), yb.to(self.device, dtype=torch.long)
            # End of epoch → reshuffle
            self.train_loader = DataLoader(
                self.full_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
            
    def get_full_data(self):
        loader = DataLoader(self.full_dataset, batch_size=len(self.full_dataset), shuffle=False)
        xb, yb = next(iter(loader))
        return xb.to(self.device), yb.to(self.device, dtype=torch.long)

    # -------------------------------------------------
    # Interface methods
    # -------------------------------------------------

    def initialize(self, param_names, model):
        self.param_names = param_names
        self.model = model

    def get_data(self):
        return next(self.train_data_gen)
         

    def get_test_data(self):
        return (self.test_images, self.test_targets)

    def predict(self, data, model):
        images, _ = data
        return model(images)  # logits

    def loss_func(self, data, y, reduction='mean'):
        _, targets = data

        # y = logits, shape (N, 10)
        loss = F.cross_entropy(y, targets, reduction=reduction)
        return {"CrossEnt": loss}

    def balanced_loss_fn(self, data, y, weights, reduction='mean'):
        loss_dict = self.loss_func(data, y, reduction=reduction)
        return loss_dict["CrossEnt"]

    def balanced_loss_fn_stateless(self, params_tuple, weights):
        param_dict = {name: p for name, p in zip(self.param_names, params_tuple)}

        images, targets = self.get_data()

        logits = functional_call(self.model, param_dict, (images,))
        loss = F.cross_entropy(logits, targets)

        return loss

    def initial_loss_weights(self):
        return {"CrossEnt": 1.0}

    def evaluate(self, model, device):
        model.eval()
        x_test, y_test = self.get_test_data()  # shape [N, 784], [N]
        x_test, y_test = x_test.to(device), y_test.to(device)

        with torch.no_grad():
            logits = model(x_test)  # [N, 10]
            pred_labels = logits.argmax(dim=1)
            accuracy = (pred_labels == y_test).float().mean().item()
            # You can also compute cross-entropy loss if you like
            loss = F.cross_entropy(logits, y_test).item()

        return {"accuracy": accuracy, "loss": loss}
