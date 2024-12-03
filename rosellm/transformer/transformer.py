import math 
import time 

import spacy
import torch
import torchtext

from torch import nn, optim 
from torch.optim import Adam 
from torch import Tensor 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


