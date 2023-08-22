from typing import Tuple
import numpy as np
import torch
import logging
from scipy.stats import ortho_group
from scipy.stats import linregress


def gen_data(datasetsize: int,
             r: int,
             device: torch.cuda.Device='cpu',
             trainsize: int=2**18,
             testsize: int=2**10,
             d: int=20,
             funcseed: int=42,
             ood: bool=False,
             normal: bool=False) -> Tuple[torch.Tensor]:
    
    ##Generate data with a true central subspaces of varying dimensions
    #generate X values for training and test sets
    if not normal:
      trainX = np.random.rand(d,trainsize).astype(np.float32)[:,:datasetsize] - 0.5 #distributed as U[-1/2, 1/2]
      testX = np.random.rand(d,testsize).astype(np.float32) - 0.5 #distributed as U[-1/2, 1/2]
    else:
      trainX = np.random.randn(d,trainsize).astype(np.float32)[:,:datasetsize] #distributed as N(0,1)
      testX = np.random.randn(d,testsize).astype(np.float32) #distributed as N(0,1)
    #out of distribution datagen
    if ood:
      trainX *= 2 #now distributed as U[-1, 1] or N(0,2)
      testX *= 2 #now distributed as U[-1, 1] or N(0,2)
    ##for each $r$ value create and store data-gen functions and $y$ evaluations
    #geneate params for functions
    k = d+1
    np.random.seed(funcseed) #set seed for random function generation
    U = ortho_group.rvs(k)[:,:r]
    Sigma = np.random.rand(r)*100
    V = ortho_group.rvs(d)[:,:r]
    W = (U * Sigma) @ V.T
    A = np.random.randn(k)
    B = np.random.randn(k)
    #create functions
    def g(z): #active subspace function
        hidden_layer = (U*Sigma)@z
        hidden_layer = hidden_layer.T + B
        hidden_layer = np.maximum(0,hidden_layer).T
        return A@hidden_layer
    def f(x): #teacher network
        z = V.T@x
        return g(z)
    #generate data
    trainY = f(trainX).astype(np.float32)
    testY = f(testX).astype(np.float32)
    #move data to device
    logging.debug("Moving train/test data to device: %s", device)
    trainX = torch.from_numpy(trainX).T.to(device)
    trainY = torch.from_numpy(trainY).to(device)
    testX = torch.from_numpy(testX).T.to(device)
    testY = torch.from_numpy(testY).to(device)
    logging.debug("trainX shape: %s, trainY shape: %s", trainX.shape, trainY.shape)

    return trainX,trainY,testX,testY
