import numpy as np
import torch
import os

from adabelief_pytorch import AdaBelief
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from antipasti.model.model import ANTIPASTI
from antipasti.utils.biology_utils import check_train_test_identity 
from config import DATA_DIR




def load_checkpoint(path, input_shape, n_filters=1, pooling_size=256, filter_size=None):
    if path.endswith('_test.pt'):
        model = ANTIPASTI(input_shape=input_shape)
        
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser = AdaBelief(model.parameters())
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    n_epochs = checkpoint['epoch']
    train_losses = checkpoint['tr_loss']
    test_losses = checkpoint['test_loss']

    
    
    return model, optimiser, n_epochs, train_losses, test_losses 
    
    