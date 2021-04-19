import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import expm

from utils import MixedDropout, sparse_matrix_to_torch


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())

def calc_LF_exact(adj_matrix: sp.spmatrix, alpha: float, mu: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = (1+ alpha * mu - alpha) * sp.eye(nnodes) + (2 * alpha - alpha * mu -1) * M
    A_outer = mu * sp.eye(nnodes) + (1 - mu) * M 
    return alpha * np.linalg.inv(A_inner.toarray()) @ A_outer

def calc_HF_exact(adj_matrix: sp.spmatrix, alpha: float, beta: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    L = sp.eye(nnodes) - M
    A_inner = alpha * sp.eye(nnodes) + (alpha * beta + 1  - alpha)  * L
    A_outer = sp.eye(nnodes) + beta * L
    return alpha * np.linalg.inv(A_inner.toarray()) @ A_outer


class PPRExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions

class LFExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, mu: float, drop_prob: float = None):
        super().__init__()

        LF_mat = calc_LF_exact(adj_matrix, alpha, mu)
        self.register_buffer('mat', torch.FloatTensor(LF_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions

class HFExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, beta: float, drop_prob: float = None):
        super().__init__()

        HF_mat = calc_HF_exact(adj_matrix, alpha, beta)
        self.register_buffer('mat', torch.FloatTensor(HF_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions


class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds
        return preds[idx]



class LFPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, mu: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter
        self.mu = mu
        # normalize self-loop A
        M = calc_A_hat(adj_matrix)
        # A_hat = 1/(1+alpha*mu-alpha) * M
        self.register_buffer('A_hat', sparse_matrix_to_torch((1/(1 + alpha * mu - alpha)) * M))  

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        # input local_preds = H
        # Z_0 = mu/(1+alpha*mu-alpha) * H + (1-mu)/(1+alpha*mu-alpha) * A @ H 
        preds = (self.mu / (1 + self.alpha * self.mu - self.alpha)) * local_preds + (1 - self.mu) * self.A_hat @ local_preds
        #residual part
        local_preds = self.alpha * preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = (1 - 2 * self.alpha + self.mu * self.alpha) * A_drop @ preds + local_preds
        return preds[idx]

class HFPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, beta: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        nnodes = adj_matrix.shape[0]
        L = sp.eye(nnodes) - M
        self.register_buffer('L_hat', sparse_matrix_to_torch(L)) # L
        self.register_buffer('A_hat', sparse_matrix_to_torch(((alpha * beta + 1 - alpha)/(alpha*beta + 1))* M)) 

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        #  Z_0 = 1/(alpha*beta + 1) H + beta/(alpha*beta + 1) LH
        preds = 1/(self.alpha * self.beta + 1) * local_preds + (self.beta/(self.alpha * self.beta + 1)) * self.L_hat  @ local_preds
        local_preds = self.alpha * preds # residual part: alpha/(alpha*beta + 1) H + alpha * beta/(alpha*beta + 1) LH
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + local_preds
        return preds[idx]
