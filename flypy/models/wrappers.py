#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 08 09:00:00 2024
@author: ike
"""


import numpy as np
import pandas as pd
import os.path as op

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score


"""
This module contains convenience wrappers around pytorch modules for model
training, evaluation, and accuracy statistic tracking.
"""


# def __call__(self):
#
#     self.decay = torch.optim.lr_scheduler.ExponentialLR(
#         optimizer=self.optim, gamma=params("Gamma"))
#
#     if checkpoint is not None:
#         self.optim.load_state_dict(checkpoint["Optimizer"])
#         self.decay.load_state_dict(checkpoint["Scheduler"])
#         epoch = checkpoint["Epoch"]
#


class RNNClassifierWrapper:
    # Transfer Data to GPU if available
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODES = ("train", "validation", "test")
    _COLS = ["mode", "epoch", "loss", "label", "prediction", "accuracy"]

    def __init__(
            self,
            model: nn.Module,
            save: str,
            criterion: type = nn.CrossEntropyLoss,
            cr_kwargs: dict = None,
            optimizer: type = torch.optim.Adam,
            op_kwargs: dict = None,
            scheduler: type = torch.optim.lr_scheduler.ReduceLROnPlateau,
            sc_kwargs: dict = None
    ):
        # set class keyword arguments
        cr_kwargs = {} if cr_kwargs is None else cr_kwargs
        op_kwargs = {} if op_kwargs is None else op_kwargs
        sc_kwargs = (
            {"factor": 0.1, "patience": 4, "threshold": 0.01}
            if sc_kwargs is None else sc_kwargs)

        # instantiate instance model and associated support
        self.model = model.double().to(self._DEVICE)
        self.criterion = criterion(**cr_kwargs)
        self.optimizer = optimizer(params=self.model.parameters(), **op_kwargs)
        self.scheduler = scheduler(
            optimizer=self.optimizer, mode="min", **sc_kwargs)
        # lr=config['lr'],weight_decay=config['weight_decay'])

        self.save = save
        self.statistics = []

    def _save_checkpoint(self, e):
        states = {
            "ml": self.model.state_dict(),
            "op": self.optimizer.state_dict(),
            "sc": self.scheduler.state_dict(),
            "statistics": self.statistics,
            "epoch": e}
        torch.save(states, self.save)

    def _load_checkpoint(self):
        if not op.isfile(self.save):
            return 0

        states = torch.load(self.save, map_location=self._DEVICE)
        self.model.load_state_dict(states["ml"])
        self.optimizer.load_state_dict(states["op"])
        self.scheduler.load_state_dict(states["sc"])
        self.statistics = states["statistics"]
        return states["epoch"]

    def _learning_mode(
            self
    ):
        self.model = self.model.train()
        torch.set_grad_enabled(True)

    def _evaluate_mode(
            self
    ):
        self.model = self.model.eval()
        torch.set_grad_enabled(False)

    def _step(
            self,
            sample,
            labels
    ):
        # forward pass, predict.shape = (batch_size, n_classes)
        predict = self.model(sample.double().to(self._DEVICE))
        labels = (
            labels if labels is None else (
                labels if torch.is_tensor(labels) else
                torch.tensor(np.atleast_1d(labels))))
        loss = None if labels is None else self.criterion(
            predict, labels.long().to(self._DEVICE))
        torch.cuda.empty_cache()
        return predict, loss

    def _clean_statistics(self):
        statistics = pd.DataFrame(self.statistics, columns=self._COLS[:-1])
        statistics[self._COLS[-1]] = statistics.apply(
            lambda x: accuracy_score(x[self._COLS[-3]], x[self._COLS[-2]]),
            axis=1)
        return statistics

    def fit(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        epochs: int,
        train_fraction: float = 0.8,
        soft=nn.Softmax(dim=1)
    ):
        train_dataset, valid_dataset = random_split(
            train_dataset, [train_fraction, 1 - train_fraction])
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        # load existing checkpoint and resume training
        e = self._load_checkpoint()

        # loop over the dataset multiple times
        for e in range(e + 1, epochs):
            r_loss = 0
            r_labels = []
            r_predictions = []

            self._learning_mode()
            for sample, labels in train_loader:
                # forward pass
                self.optimizer.zero_grad()
                predict, loss = self._step(sample, labels)

                # back propagation
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1, 2)
                self.optimizer.step()

                r_loss += loss.item()
                r_labels += [labels.cpu().detach().numpy()]
                r_predictions += [
                    np.argmax(soft(predict).cpu().detach().numpy(), axis=-1)]

            # compute and save epoch training statistics
            r_loss = r_loss / len(train_dataset)
            r_labels = np.concatenate(r_labels, axis=0)
            r_predictions = np.concatenate(r_predictions, axis=0)
            self.statistics += [
                [self._MODES[0], e, r_loss, r_labels, r_predictions]]

            # compute and save validation statistics
            v_loss, v_labels, v_predictions = self.predict(
                valid_dataset, batch_size, soft)
            self.statistics += [
                [self._MODES[1], e, v_loss, v_labels, v_predictions]]

            # modify learning rate based on validation loss
            self.scheduler.step(r_loss)
            self._save_checkpoint(e)

        # compute and save test statistics
        t_loss, t_labels, t_predictions = self.predict(
            test_dataset, batch_size, soft)
        self.statistics += [
            [self._MODES[2], e, t_loss, t_labels, t_predictions]]
        return self._clean_statistics()

    def predict(
            self,
            test_dataset: Dataset,
            batch_size: int,
            soft=nn.Softmax(dim=1),
    ):
        r_loss = 0.0
        r_labels = []
        r_predictions = []
        self._evaluate_mode()

        # loop over dataset once
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for sample, labels in loader:
            predict, loss = self._step(sample, labels)
            r_loss += 0 if loss is None else loss.item()
            r_labels += [
                None if labels is None else labels.cpu().detach().numpy()]
            r_predictions += [
                np.argmax(soft(predict).cpu().detach().numpy(), axis=-1)]

        r_loss = r_loss / len(test_dataset)
        r_labels = (
            [None] if r_labels[0] is None else
            np.concatenate(r_labels, axis=0))
        r_predictions = np.concatenate(r_predictions, axis=0)
        return r_loss, r_labels, r_predictions
