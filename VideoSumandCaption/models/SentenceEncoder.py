import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CustomSentenceEncoder(nn.Module):
    def __init__(self, device, emb, word_dim, num_layer, hidden):
        super(CustomSentenceEncoder, self).__init__()
        self.emb = emb
        self.word_dim = word_dim
        self.hidden = hidden
        self.device = device
        self.lstm = nn.LSTM(word_dim, hidden, num_layers=num_layer, batch_first=True)

    def forward(self, inp):
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp).to(self.device)
        inp = self.emb(inp)
        out, _ = self.lstm(inp)
        return out

