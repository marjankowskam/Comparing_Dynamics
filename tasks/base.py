from abc import ABC, abstractmethod

class Task(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def get_full_data(self):
        return self.get_data()

    def get_data(self):
        pass

    def get_test_data(self):
        # return a tuple of data and test_targets
        pass

    def initialize(self, param_names, model):
        # only if needed
        pass

    def loss_func(self, data, y):
        # should return a dictionary indexed by loss component names
        pass

    def balanced_loss_fn(self, data, y, weights):
        # weight loss components
        pass

    def predict(self, data, model): 
        # predictions for the input data 
        pass
    
    def balanced_loss_fn_stateless(self, params_tuple, weights):
        # stateless loss function expected by the balancer
        # only do functional calls of the model with the given fixed parameters
        pass

    def initial_loss_weights(self):
        # return a dictionary of initial loss balancer values - keys are the loss names
        pass
    
    @abstractmethod
    def evaluate(self, model, device):
        """
        Evaluate the model on test data and return a dictionary of metrics.
        Each Task can decide which metrics to compute.
        """
        pass
    