import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FS

class DChord(nn.Module):
    """ Model for template-based chord recoginition. performs thresholded 12 normalization of the input features
    followed by template matching (convolution) with a set of chord templates. The chord similarities are max normalized afterwards.

    Args:
        chord_templates_norm: tensor of chord templates to be used: either of shape (24, 12) or (25, 12)
        norm_threshold: threshold to be used for normalization: vectors below the threshold are replaced by the unit vector of the repesctive norm.
    """

    def __init__(self, chord_templates_norm, norm_threshold=1e-4):
        super().__init__()
        self.norm_threshold = norm_threshold
        self.filter = torch.nn.Conv2d(in_channels=1, out_channles=chord_templates_norm.shape[0],
                                      kernel_size=(1, 12), stride=(1, 1), bias=False)
        # set weights of convolutional layer
        self.filter.weight = torch.unsqueeze(torch.unsqueeze(chord_templates_norm, 1), 1)
    
    def forward(self, x):
        # x must be of shape (B x 1 x T x 12) 
        
        # thresholded l2 normalization of features
        x_norms = torch.linalg.vector_norm(x, ord=2.0, dim=3)                                         
        idx = x_norms > self.norm_threshold
        x_normalized = F.normalize(x.clone(), p=2.0, dim=3, eps=self.norm_threshold)
        x_normalized[~idx] = torch.ones(12, dtype=torch.float32) / torch.sqrt(torch.tensor([12]))    # out: (B x 1 x T x 12)
        
        # template matching
        x_chord_sim = self.filter(x_normalized)    # out: (B x 24/25 x T x 1)
        
        # move chord similarities to last axis
        x_chord_sim = torch.swapaxes(x_chord_sim, 1, 3)    # out: (B x 1 x T x 24/25)
        
        # apply max norm
        x_chord_sim_norms = torch.linalg.vector_norm(x_chord_sim, ord=float('inf'), dim=3)
        idx_sim = x_chord_sim_norms > self.norm_threshold
        x_chord_sim_normalized = F.normalize(x_chord_sim.clone(), p=float('inf'), dim=3, eps=self.norm_threshold)
        x_chord_sim_normalized[~idx_sim] = torch.ones(x_chord_sim_normalized.shape[3], dtype=torch.float32)
        return x_chord_sim_normalized    # out: (B x 1 x T x 24/25)