import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn_models.nn_processing import log_compression, temporal_smoothing, feature_normalization, softmax_temperature

class dchord_templates(torch.nn.Module):
    """Module for applying template-based chord recognition to chroma features.

    Args:
        shared_weights: Whether to use 2 shared kernels (maj/min) or 24 individual chord templates
        initialize_parameters: Whether to initialize kernels with idealized binary chord templates and zero bias
        normalize_weights: Whether to normalize all templates to unit Euclidean norm
        bias: Whether to allow for a trainable bias
    """
    def __init__(self, shared_weights=False, initialize_parameters=True, normalize_weights=True, bias=False):
        super(dchord_templates, self).__init__()

        if shared_weights:
            self.padding = lambda x: F.pad(x, pad=(0, 11, 0, 0), mode='circular')
            self.filter = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 12), stride=(1, 1), bias=bias)
            if initialize_parameters:
                self.filter.weight.data = torch.tensor([[[[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]],     # major
                                                        [[[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]]]],    # minor
                                                       dtype=torch.float32)
                if self.filter.bias is not None:
                    self.filter.bias.data = torch.zeros_like(self.filter.bias.data, dtype=torch.float32)

        else:
            self.padding = lambda x: x
            self.filter = torch.nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(1, 12), stride=(1, 1), bias=bias)
            if initialize_parameters:
                kernel_major = torch.tensor([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
                kernel_minor = torch.tensor([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

                for i in range(12):
                    self.filter.weight.data[i, 0, 0, :] = torch.roll(kernel_major, i)
                    self.filter.weight.data[12 + i, 0, 0, :] = torch.roll(kernel_minor, i)

                if self.filter.bias is not None:
                    self.filter.bias.data = torch.zeros_like(self.filter.bias.data, dtype=torch.float32)

        if normalize_weights:
            self.filter.weight.data = F.normalize(self.filter.weight.data, p=2.0, dim=3)

    def forward(self, x):
        x_padded = self.padding(x)                                          # out: (B x 1 x T x 23) or (B x 1 x T x 12)
        y = self.filter(x_padded)                                           # out: (B x 2 x T x 12) or (B x 24 x T x 1)
        y_reshaped = torch.swapaxes(y, 1, 2)                                # out: (B x T x 2 x 12) or (B x T x 24 x 1)
        y_reshaped = torch.unsqueeze(torch.flatten(y_reshaped, start_dim=2), 1)             # out: (B x 1 x T x 24)
        return y_reshaped
    

class dchord_pipeline(torch.nn.Module):
    """ Model for template-based chord recognition:
    
    Pipeline:
        1. Input
        2. Log Compression
        2.5 Temporal Smoothing
        4. Normalization
        5. Chord Recognition via Templates
        6. Softmax -> output: chord probabilities
        
    Args:
        dictionaries containing parameters for the individual building blocks
    
    """
    
    def __init__(self, compression_params=None, temp_smooth_params=None, feature_norm_params=None,
                 chord_template_params=None, softmax_params=None):
        super(dchord_pipeline, self).__init__()

        self.log_compression = log_compression(**compression_params)
        # self.temporal_smoothing = temporal_smoothing(**temp_smooth_params)
        self.feature_normalization = feature_normalization(**feature_norm_params)
        self.chord_template_params = dchord_templates(**chord_template_params)
        self.softmax_temperature = softmax_temperature(**softmax_params)

    def forward(self, x):
        y_pred, _, _, _, _ = self.get_intermediate_data(x)
        return y_pred
    
    def get_intermediate_data(self, x):
        x_comp = self.log_compression(x)
        # No need for temporal smoothing in the current implementation
        x_avg = x_comp
        
        x_norm = self.feature_normalization(x_avg)
        x_templates = self.chord_template_params(x_norm)
        y_pred = self.softmax_temperature(x_templates)
        return y_pred, x_templates, x_norm, x_avg, x_comp
    
    def fit(self, x, y):
        """Fit the model to the data."""
        pass
    
