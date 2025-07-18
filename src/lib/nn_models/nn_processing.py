import torch
import torch.nn.functional as F
import numpy as np


class log_compression(torch.nn.Module):
    """Module for logarithmic compression of an array

    Args:
        gamma: Compression factor
        trainable: Whether the gradient w.r.t. gamma is computed in backward pass
    """
    def __init__(self, gamma_init=1, trainable=True):
        super(log_compression, self).__init__()

        # define logarithm of gamma as trainable parameter
        if gamma_init is not None:
            self.log_gamma = torch.nn.parameter.Parameter(data=torch.log(torch.as_tensor(gamma_init, dtype=torch.float32)), requires_grad=trainable)
        else:
            self.log_gamma = None

    def forward(self, x):
        if self.log_gamma is not None:
            return torch.log(1.0 + torch.exp(self.log_gamma) * x)
        else:
            return x


class gaussian_filter(torch.nn.Module):
    """Module for generating a 1D Gaussian filter

    Args:
        length: kernel length
        sigma: (initial) standard deviation for Gaussian kernel
        dim: dimension across which to apply the filter; output tensor will have singleton dimension at 'dim'
        trainable: whether to optimize the standard deviation
    """
    def __init__(self, length=41, sigma_init=1, dim=2, trainable=True):
        super(gaussian_filter, self).__init__()

        self.length = length
        self.dim = dim

        if sigma_init is None:
            sigma_init = length
        self.log_sigma = torch.nn.parameter.Parameter(data=torch.log(torch.as_tensor(sigma_init, dtype=torch.float32)), requires_grad=trainable)

    def forward(self, x):
        idx = x.ndim * [1]
        idx[self.dim] = self.length
        w = self.get_kernel()
        x_smoothed = torch.sum(x * w.view(*idx), dim=self.dim, keepdim=True)
        return x_smoothed

    def get_kernel(self):
        n = torch.arange(0, self.length).to(self.log_sigma.device) - (self.length - 1.0) / 2.0
        sig2 = 2 * torch.exp(self.log_sigma) ** 2
        w = torch.exp(-n ** 2 / sig2)
        return w / torch.sum(w)


class temporal_smoothing(torch.nn.Module):
    """Module for temporal smoothing of a feature sequence.

    Args:
        smoothing_type: Either 'weighted_sum', 'median' or 'Gaussian'
        avg_length: Length to be averaged over; only relevant for 'weighted_sum' and 'Gaussian' (median is taken over whole input length)
        weight_init: How to initialize (trainable) weights, only relevant for 'weighted_sum' (either 'uniform' or 'random')
    """
    def __init__(self, smoothing_type='weighted_sum', avg_length=41, weight_init='uniform', sigma_init=20):
        super(temporal_smoothing, self).__init__()
        if smoothing_type not in {'weighted_sum', 'median', 'Gaussian'}:
            raise ValueError('Smoothing type ' + smoothing_type + ' is unknown!')

        if weight_init not in {'random', 'uniform'}:
            raise ValueError('Weight initialization ' + weight_init + ' is unknown!')

        self.avg_length = avg_length

        if smoothing_type == 'weighted_sum':
            self.filter = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(avg_length, 1),
                                    stride=(1, 1), bias=False, padding='valid')

            if weight_init == 'random':
                pass
            elif weight_init == 'uniform':
                self.filter.weight.data = torch.ones_like(self.filter.weight.data) / avg_length

        elif smoothing_type == 'median':
            # Median operator don't have a gradient, so it is not trainable
            self.filter = lambda x: torch.median(x, dim=2, keepdim=True,).values

        elif smoothing_type == 'Gaussian':
            self.filter = gaussian_filter(length=avg_length, sigma_init=sigma_init, dim=2, trainable=True)

    def forward(self, x):
        # treat channels as batch dimension to use conv. layer with 1 channel
        x_reshaped = x.view(-1, 1, *x.shape[2:])
        x_filtered = self.filter(x_reshaped)
        return x_filtered.view(*x.shape[:2], *x_filtered.shape[2:])


class feature_normalization(torch.nn.Module):
    """Module for feature normalization

    Args:
        num_features: Number of features
        norm: The norm to be applied. '1', '2', 'max' or 'z'
        threshold: Threshold below which the vector `v` is used instead of normalization
        v: Used instead of normalization below `threshold`. If None, uses unit vector for given norm
    """
    def __init__(self, num_features=12, norm='2', threshold=1e-4, v=None, dim=3):
        super(feature_normalization, self).__init__()
        if norm not in ['1', '2', 'max', 'z']:
            raise ValueError('Norm ' + norm + ' is unknown!')

        self.threshold = threshold
        self.v = v

        if norm == '1':
            if self.v is None:
                self.v = torch.ones(num_features, dtype=torch.float32) / num_features
            self.get_norms = lambda x: torch.linalg.vector_norm(x, ord=1.0, dim=dim, keepdim=False)
            self.normalize = lambda x: F.normalize(x, p=1.0, dim=dim, eps=threshold)

        if norm == '2':
            if self.v is None:
                self.v = torch.ones(num_features, dtype=torch.float32) / torch.sqrt(torch.tensor([num_features]))
            self.get_norms = lambda x: torch.linalg.vector_norm(x, ord=2.0, dim=dim, keepdim=False)
            self.normalize = lambda x: F.normalize(x, p=2.0, dim=dim, eps=threshold)

        if norm == 'max':
            if self.v is None:
                self.v = torch.ones(num_features, dtype=torch.float32)
            self.get_norms = lambda x: torch.linalg.vector_norm(x, ord=float('inf'), dim=dim, keepdim=False)
            self.normalize = lambda x: F.normalize(x, p=float('inf'), dim=dim, eps=threshold)

        if norm == 'z':
            if self.v is None:
                self.v = torch.zeros(num_features)
            self.get_norms = lambda x: torch.std(x, dim=dim, keepdim=False, unbiased=True)
            self.normalize = lambda x: (x - torch.mean(x, dim=dim, keepdim=True)) / torch.std(x, dim=dim, keepdim=True,
                                                                                              unbiased=True)
    def forward(self, x):
        x_norms = self.get_norms(x)
        idx = x_norms > self.threshold
        x_normalized = x.clone()
        x_normalized = self.normalize(x_normalized)
        x_normalized[~idx] = self.v.to(x.device)
        return x_normalized

class softmax_temperature(torch.nn.Module):
    """Softmax activation with trainable temperature parameter

    Args:
        dim: Dimension across which to apply the softmax function
        tau: Temperature parameter
        trainable: Whether the gradient w.r.t. tau is computed in backward pass
    """
    def __init__(self, dim=3, tau=1, trainable=True):
        super(softmax_temperature, self).__init__()
        self.dim = dim
        self.tau = torch.nn.parameter.Parameter(data=torch.tensor([tau], dtype=torch.float32), requires_grad=trainable)

    def forward(self, x):
        return F.softmax(x / self.tau, dim=self.dim)