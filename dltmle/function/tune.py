import os
import datetime

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import lightning.pytorch as pl
import optuna

from ..model.dltmle import DeepLTMLE
from ..utils import seed_everything, get_torch_device

class DatasetWLACY(Dataset):
    def __init__(self, W, L, A, C, Y, dtype=torch.float32):
        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.C = torch.tensor(C, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        
        self.dim_static = W.shape[1]
        self.dim_dynamic = L.shape[2]
        self.tau = L.shape[1]
        
    def __len__(self):
        return self.W.shape[0]
    
    def __getitem__(self, index):
        return {
            'W': self.W[index],
            'L': self.L[index],
            'A': self.A[index],
            'C': self.C[index],
            'Y': self.Y[index],
            'a': self.A[index],
        }

def tune(rng_seed, hparam_candidates, W, L, A, C, Y, n_trials=100, log_hparams=True):
    '''fit the model using the training data
    
    Parameters
    ----------
    hparams: `dict`
        hyperparameters for the model
    W: `torch.Tensor`
        static covariates
    L: `torch.Tensor`
        dynamic covariates
    A: `torch.Tensor`
        treatment
    C: `torch.Tensor`
        censoring
    Y: `torch.Tensor`
        outcome
    a: `torch.Tensor`
        counterfactual treatment
    
    Returns
    -------
    model: `dict`
        fitted model
    '''
    # set the seed
    seed_everything(rng_seed)
    
    dataset = DatasetWLACY(W, L, A, C, Y)
    train, val = random_split(dataset, [0.8, 0.2])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def objective(trial: optuna.trial.FixedTrial):
        hparams = {key: trial.suggest_categorical(key, value) for key, value in hparam_candidates.items()}
        
        # initialize the model
        model = DeepLTMLE(
            dataset.dim_static, dataset.dim_dynamic, dataset.tau,
            **hparams
            )
        
        logger = pl.loggers.TensorBoardLogger(
            save_dir=os.path.join('artifact', 'tune'),
            name=timestamp,
            default_hp_metric=False,
            )
        
        if log_hparams:
            logger.log_hyperparams(
                hparams,
                metrics={'val/L':0, 'val/G':0, 'val/Q':0, 'val/GQ':0, 'val/Q_star':0}
                )
        
        trainer = pl.Trainer(
            max_epochs=hparams['max_epochs'],
            accelerator=get_torch_device(),
            logger=logger,
            )
        
        train_loader = DataLoader(train, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val, batch_size=hparams['batch_size'], shuffle=False)
        
        trainer.fit(model, train_loader, val_loader)
        
        return trainer.logged_metrics['val/GQ']
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.NSGAIISampler(seed=np.random.randint(0, 2**32)),
        )
    
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params
        
    
    
