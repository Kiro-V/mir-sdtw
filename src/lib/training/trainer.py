import os
import sys
import copy
from typing import Self

import torch as t
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import logging
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self,
                 model,
                 crit,
                 optimizer=None,
                 train_dl=None,
                 val_dl=None,
                 training_param=None,
                 save_e_matrix_to=False,
                 save_e_matrix_params={'save_path': 'e_matrix', 'every_n_epochs': 1},
                 plot_loader=None,
                 plot_loader_strong=None,
                 ):
        """ Train a neural network model (torch.nn.Module)

            Args:
                model: model to be trained
                crit: loss function
                optimizer: optimizerizer to be used
                train_dl: training data loader
                val_dl: validation data loader
                device: device to use for training (default: 'cuda:0')
                only_save_best: whether to only save best trainer ckp / model and thus overwrite previous best states
                verbose: whether to print training stats (default: True)
        """
        self.__model = model
        self.__crit = crit
        self.__optimizer = optimizer
        self.__train_dl = train_dl
        self.__val_dl = val_dl
        self.__device = training_param['device'] if training_param is not None else 'cpu'
        self.__split_weak_batch = training_param['split_weak_batch'] if training_param is not None else False
        self.__epochs = training_param['max_epochs'] if training_param is not None else 100

        self.__save_e_matrix_to = save_e_matrix_to
        self.__save_e_matrix_params = save_e_matrix_params
        self.__plot_loader = plot_loader
        self.__plot_loader_strong = plot_loader_strong

        self._train_losses, self._val_losses = [], []


    def __save_e_matrix(self):
        """
        Save E matrix plot for monitoring training progress
        Perform the validation step and plot the E matrix
        """
        if not self.__plot_loader:
            raise ValueError("Plot loaders must be provided for plotting E matrix.")
        
        if not self.__plot_loader_strong:
            raise ValueError("Strong plot loader must be provided for obtaining ground truth E matrix.")
        
        if not os.path.exists(self.__save_e_matrix_params['save_path']):
            os.makedirs(self.__save_e_matrix_params['save_path'])
            logging.info(f"Created directory for E matrix plots: {self.__save_e_matrix_params['save_path']}")
        
        if not self.__save_e_matrix_params['every_n_epochs']:
            # Default to saving every epoch
            self.__save_e_matrix_params['every_n_epochs'] = 1

        self.__model.eval()
        x_weak, y_weak = next(iter(self.__plot_loader))
        y_weak_pred = self.__model(x_weak.to(self.__device))
        # Squeeze channel dimension
        y_weak_pred = torch.squeeze(y_weak_pred, 1)
        y_weak = torch.squeeze(y_weak, 1)
        loss_ = self.__crit(y_weak_pred.to(self.__device), y_weak.to(self.__device))
        loss_.mean().item()
        loss_.backward()
        e_matrix_soft = self.__crit.dtw_class.e_matrix.cpu().detach().numpy()

        # Obtaining the ground truth alignment E matrix
        if not self.__plot_loader_strong:
            raise ValueError("Strong plot loader must be provided for obtaining ground truth E matrix.")
        x_strong, y_strong = next(iter(self.__plot_loader_strong))
        print(f"x_strong: {x_strong.shape}, y_strong: {y_strong.shape}")
        y_strong = torch.squeeze(y_strong, 1)
        # Ground truth region alignment matrix
        e_matrix_strong = np.dot(y_strong[0,:,:].cpu().numpy(), torch.permute(y_weak[0,:,:], (1, 0)).cpu().numpy())
        plt.figure(figsize=(10, 6))
        plt.imshow(e_matrix_soft[0,:,:].T, cmap='gray_r', origin='lower', aspect='auto')
        plt.colorbar(label='Probability')
        plt.imshow(e_matrix_strong.T, cmap='Reds', origin='lower', alpha=0.25, aspect='auto')
        plt.legend(['Ground Truth Alignment', 'Predicted Alignment'])
        plt.ylabel('Soft Sequences', fontsize=18)
        plt.xlabel('Strong Sequences', fontsize=18)
        plt.title('E Matrix', fontsize=18)
        plt.savefig(os.path.join(self.__save_e_matrix_params['save_path'], f"e_matrix_epoch_{len(self._train_losses)}.png") )
        plt.close()

    def __train_step(self,x, y):
        """
        Training single mini-batch

        """
        self.__model.zero_grad()
        y_pred = self.__model(x)

        # Squeeze channel dimension
        y_pred = torch.squeeze(y_pred, 1)
        y = torch.squeeze(y, 1)

        if self.__split_weak_batch:
            batch_size = y_pred.size(0)
            loss = 0
            for b in range(batch_size):
                loss += self.__crit(y_pred[b], y[b])
            loss /= batch_size # average loss over batch_size
        else:
            loss = self.__crit(y_pred, y)

        loss.mean().backward()
        self.__optimizer.step()
        return loss.mean().item()
    
    def __val_test_step(self, x, y):
        """
        Validation step for a single mini-batch
        """
        with t.no_grad():
            y_pred = self.__model(x)

            # Squeeze channel dimension
            y_pred = torch.squeeze(y_pred, 1)
            y = torch.squeeze(y, 1)

            if self.__split_weak_batch:
                batch_size = y_pred.size(0)
                loss = 0
                for b in range(batch_size):
                    loss += self.__crit(y_pred[b], y[b])
                loss /= batch_size
            else:
                print(f"y_pred: {y_pred.shape}, y: {y.shape}"   )
                loss = self.__crit(y_pred, y)
            return loss.mean().item(), y_pred

    def __train_epoch(self, epoch=0):
        """
        Train the model for one epoch
        """
        self.__model.train()
        train_loss = 0.0
        for i, (x, y) in enumerate(self.__train_dl):
            x, y = x.to(self.__device), y.to(self.__device)
            loss = self.__train_step(x, y)
            train_loss += loss
            if i % 10 == 0:
                logging.info(f'Epoch {epoch}, Step {i}, Loss: {loss:.4f}')
        return train_loss / len(self.__train_dl)

    def _validate(self):
        """
        Computes and return validation loss using validation data loader
        """
        self.__model.eval()

        val_loss = 0.0

        with t.no_grad():
            for i, (x, y) in enumerate(self.__val_dl):
                x, y = x.to(self.__device), y.to(self.__device)
                loss, _ = self.__val_test_step(x, y)
                val_loss += loss
        return val_loss / len(self.__val_dl)

    def _test(self, test_dl):
        """
        Computes and return test loss using test data loader
        """
        self.__model.eval()

        test_loss = 0.0
        with t.no_grad():
            for i, (x, y) in enumerate(test_dl):
                x, y = x.to(self.__device), y.to(self.__device)
                loss, _ = self.__val_test_step(x, y)
                test_loss += loss
        return test_loss / len(test_dl)
    
    def fit(self):
        """
        Fit the model to the training data
        Returns:
            train_loss: List of training losses for each epoch
            val_loss: List of validation losses for each epoch

        """
        best_val_loss = None
        best_model = None
        
        for e in range(self.__epochs):
            train_loss = self.__train_epoch(e)
            self._train_losses.append(train_loss)

            val_loss = self._validate()
            self._val_losses.append(val_loss)

            print(f'Epoch {e+1}/{self.__epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if self.__save_e_matrix_to and e % self.__save_e_matrix_params['every_n_epochs'] == 0:
                self.__save_e_matrix()
            
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.__model)
                best_epoch = e
                print(' ....saved model')

        print(f'Best model found at epoch {best_epoch} with validation loss {best_val_loss:.4f}')
        print(f'\nRestoring best model...')
        self.__model = best_model
        return best_model, self._train_losses, self._val_losses

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model.to(self.__device)