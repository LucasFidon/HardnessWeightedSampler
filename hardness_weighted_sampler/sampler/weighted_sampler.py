import torch
from torch.utils.data import Sampler
import torch.nn.functional as F
import numpy as np


class WeightedSampler(Sampler):
    """
    PyTorch implementation for Hardness Weighted Sampler.

    The weighted sampler allows to sample examples in a dataset
    with respect to a custom distribution computed as:
        \f$
        distrib = \softmax(\beta \textup{weights})
        \f$
    There is one weight per example in the dataset.
    The weights can be updated dynamically during training.
    """
    def __init__(self, beta=100, weights_init=1, num_samples=-1, momentum=0.):
        """
        :param beta: float; robustness parameter (must be positive).
        It allows to interpolate between empirical risk minimization (beta=0),
        and worst case minimization (beta=+inf).
        :param weights_init: int, float or 1d tensor;
        initialization for the weights of the sampling.
        A good choice of constant for the initialization is the maximum value
        of the loss function used. Its size should be equal to num_samples.
        :param num_samples: int; number of samples in the dataset.
        num_samples must be specified if weights_init is a constant value.
        When weights_init is a 1d tensor, the number of samples is inferred
        automatically from the size of weights_init.
        """
        self.num_samples = num_samples
        # Distributionally robustness parameter
        self.beta = beta

        # Momentum used for the update of the loss history.
        self.momentum = momentum

        # Initialization of the per-example loss values with a constant value
        if isinstance(weights_init, float) or isinstance(weights_init, int):
            assert num_samples > 0, \
                "The number of samples should be specified if a constant weights_init is used"
            print('Initialize the weights of the hardness weighted sampler to the value', weights_init)
            # Add some gaussian noise on top of the initial weight value.
            # This is used to break symmetry at the initialization.
            self.weights = torch.tensor(
                np.random.normal(loc=weights_init, scale=0.001*weights_init, size=num_samples))
        # Initialization with a given vector of per-example loss values
        else:
            if isinstance(weights_init, np.ndarray):  # Support for numpy arrays
                weights_init = torch.tensor(weights_init)
            assert len(weights_init.shape) == 1, "initial weights should be a 1d tensor"
            self.weights = weights_init.float()
            if self.num_samples <= 0:
                self.num_samples = weights_init.shape[0]
            else:
                assert self.num_samples == weights_init.shape[0], \
                    "weights_init should have a size equal to num_samples"

    def get_distribution(self):
        # Apply softmax to the weights vector.
        # This seems to be the most numerically stable way
        # to compute the softmax
        distribution = F.log_softmax(
            self.beta * self.weights, dim=0).data.exp()
        return distribution

    def draw_samples(self, n):
        """
        Draw n sample indices using the hardness weighting sampling method.
        """
        eps = 0.0001 / self.num_samples
        # Get the distribution (softmax)
        distribution = self.get_distribution()
        p = distribution.numpy()
        # Set min proba to epsilon for stability
        p[p <= eps] = eps
        p /= p.sum()
        # Use numpy implementation of multinomial sampling because it is much faster
        # than the one in PyTorch
        sample_list = np.random.choice(
            self.num_samples,
            n,
            p=p,
            replace=False,
        ).tolist()
        return sample_list

    def get_importance_sampling_weights(self, batch_new_weights, batch_indices):
        # Must be called before updating the weights
        assert len(batch_indices) == batch_new_weights.size()[0], "number of weights in " \
                                                               "input batch does not " \
                                                               "correspond to the number " \
                                                               "of indices."
        log_importance_weights = []
        for idx, new_weight in zip(batch_indices, batch_new_weights):
            w = self.beta * (1. - self.momentum) * (new_weight - self.weights[idx])
            log_importance_weights.append(w)
        importance_weights = torch.tensor(
            log_importance_weights, requires_grad=False).exp().float()
        return importance_weights

    def update_weight(self, idx, new_weight):
        """
        Update the weight of sample idx for new_weight.
        :param idx: int; index of the sample of which the weight have to be updated.
        :param new_weight: float; new weight value for idx.
        """
        # Modify this function for other update strategies.
        self.weights[idx] = self.momentum * self.weights[idx] \
                            + (1. - self.momentum) * new_weight

    def save_weights(self, save_path):
        torch.save(self.weights, save_path)

    def load_weights(self, weights_path):
        print('Load the sampling weights from %s' % weights_path)
        weights = torch.load(weights_path)
        self.weights = weights
        self.num_samples = self.weights.size()[0]

    def hardest_samples_indices(self, num=100):
        """
        Return the indices of the samples with the highest loss.
        :param num: int; number of indices to return.
        :return: int list; list of indices.
        """
        weights_np = np.array(self.weights)
        hardest_indices = np.argsort((-1) * weights_np)
        return hardest_indices[:num].tolist()

    def __iter__(self):
        sample_list = self.draw_samples(self.num_samples)
        return iter(sample_list)

    def __len__(self):
        return self.num_samples
