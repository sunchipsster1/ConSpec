from misc_util import orthogonal_init, xavier_uniform_init
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from a2c_ppo_acktr.utils import init
import traceback

np.set_printoptions(threshold=10_000)
torch.set_printoptions(threshold=10_000)

class moduleCL(nn.Module):
    def __init__(self,
                 input_size, hidden_size, head, device):
        super(moduleCL, self).__init__()
        self.head = head
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = [None] * head
        self.device = device

        self.main = nn.ModuleList([nn.Sequential(
            (nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)), nn.ReLU(),
        ) for i in range(head)])
        self.layers = nn.ModuleList([nn.Sequential(
            (nn.Linear(in_features=hidden_size, out_features=hidden_size,bias=False)), nn.ReLU(),
        ) for i in range(head)])

        self.layers2 = nn.ParameterList([nn.Parameter(
            torch.randn(1, self.hidden_size) * 1.) for i
            in range(head)])

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, hidden, reward, keyused, obs, time, seed, doimage):  # obs =  [600, 16, 1, 54, 64
        #################ConSpec module is implemented here
        keysUsedOrig = keyused
        out1 = [None] * self.head
        cossim2 = [None] * self.head
        ortho2 = [None] * self.head
        minibatchsize = hidden.shape[1]
        miniSuccesssize = int(hidden.shape[1] / 2)

        for iii in range(self.head):
            out1[iii] = self.layers[iii](self.main[iii](hidden))  # hidden = length, minibatch, latent       #keyhead
            s1, s2, s3 = out1[iii].shape
            currentweights = self.layers2[iii].reshape(1, 1, -1).repeat(s1, 1, 1)
            cossim2[iii] = self.cos(out1[iii], currentweights)
            ortho2[iii] = F.softmax(cossim2[iii].squeeze() * 100., dim=0)  # length,minibatch,1 --> length,minibatch

        cossimtotal = torch.stack(cossim2,
                                  dim=2)
        orthototal = torch.stack(ortho2, dim=2)  # length,minibatch,keys

        cossimtotalmaxxx, indmaxes = (
            torch.max(cossimtotal, dim=0))
        cossimtotalmax = cossimtotalmaxxx.squeeze()  # minibatch,keys

        # '''
        costFit = ((torch.abs(1 - cossimtotalmax[:miniSuccesssize]).mean(0) + torch.abs(
            cossimtotalmax[miniSuccesssize:]).mean(0)).squeeze()) * 1.
        if keyused > -0.5:
            costFitFinal = costFit[keysUsedOrig]
        else:
            costFitFinal = costFit.sum()
        orthototalperm = orthototal.permute(1, 0, 2)
        orthototalperm = F.normalize(orthototalperm, dim=1, p=2)
        cosnorm = torch.abs(torch.matmul(orthototalperm.permute(0, 2, 1), orthototalperm))
        cosnormdiag = torch.diagonal(cosnorm, dim1=1, dim2=2)
        orthogonality = (cosnorm[:, :, :])[:miniSuccesssize] - torch.diag_embed(cosnormdiag[:, :], dim1=1, dim2=2)[
                                                               :miniSuccesssize]
        orthogonalitycost = orthogonality.sum()

        costfinal = costFitFinal * 1. + (orthogonalitycost * .2) /self.head
        pos = torch.abs(cossimtotalmax[:miniSuccesssize]).mean(0)
        neg = torch.abs(cossimtotalmax[miniSuccesssize:]).mean(0)

        return costfinal, cossimtotal, [pos, neg]


