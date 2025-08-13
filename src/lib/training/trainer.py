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
                 plot_e_matrix=False,
                 plot_e_matrix_params=None,
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

        self.__plot_e_matrix = plot_e_matrix
        self.__plot_e_matrix_params = plot_e_matrix_params
        self.__plot_loader = plot_loader
        self.__plot_loader_strong = plot_loader_strong

        self._train_losses, self._val_losses = [], []


    def _plot_e_matrix(self):
        # WIP
        x = self.plot_e_matrix_params["x"]
        y = self.plot_e_matrix_params["y"]
        lastChange = self.plot_e_matrix_params["lastChange"]
        opt_paths = self.plot_e_matrix_params["opt_paths"]
        
        self._model.zero_grad()
        
        fig, ax = plt.subplots(4, 4, figsize=(20, 10))
        ax = ax.flatten()

        for b in range(y.shape[0]):
            # forward pass
            y_pred = self._model(x[b:b+1,:])

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

            if self.__plot_e_matrix:
                self._plot_e_matrix()
            
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