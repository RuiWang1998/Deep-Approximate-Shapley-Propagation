import torch
import torch.nn as nn
import torch.nn.functional as F



class DASP(object):
    def __init__(self, model:nn.Module):
        self.model = model