import os
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from ..model.dltmle import DeepLTMLE
from ..ral import RALEstimate
from ..utils import seed_everything, get_torch_device

class DatasetWLACYa(Dataset):
    def __init__(self, W, L, A, C, Y, a, dtype=torch.float32):
        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.C = torch.tensor(C, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.a = torch.tensor(a, dtype=dtype)
        
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
            'a': self.a[index],
        }

def fit(rng_seed, hparams, W, L, A, C, Y, a):
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
    
    dataset = DatasetWLACYa(W, L, A, C, Y, a)
        
    # initialize the model
    model = DeepLTMLE(
        dataset.dim_static,
        dataset.dim_dynamic,
        dataset.tau,
        **hparams
        )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join('artifact', 'fit'),
        name='dltmle',
        sub_dir=timestamp
        )
    
    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator=get_torch_device(),
        logger=logger,
        )
    
    loader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True)
    trainer.fit(model, loader)
    
    loader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=False)
    preds = trainer.predict(model, loader)
    
    est, se, ic = model.get_estimates_from_prediction(preds, loader)
    
    return RALEstimate(est, ic)
    
    
