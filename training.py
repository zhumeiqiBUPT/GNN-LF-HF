from typing import Type, Tuple
import time
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
from earlystopping import EarlyStopping, stopping_args
from utils import matrix_to_torch
from sparsegraph import SparseGraph


def gen_seeds(size: int = None) -> np.ndarray:
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(max_uint32 + 1, size=size, dtype=np.uint32)


def get_dataloaders(idx, labels_np, batch_size=None):
    labels = torch.LongTensor(labels_np)
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(ind, labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                   for phase, dataset in datasets.items()}
    return dataloaders


def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm



def train_model(
        idx_np, name: str, model_class: Type[nn.Module], graph: SparseGraph, model_args: dict,
        learning_rate: float, reg_lambda: float,
        stopping_args: dict = stopping_args,
        test: bool = True, device: str = 'cuda',
        torch_seed: int = None, print_interval: int = 10) -> Tuple[nn.Module, dict]:

    labels_all = graph.labels
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}

    logging.log(21, f"{model_class.__name__}: {model_args}")
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed)
    logging.log(22, f"PyTorch seed: {torch_seed}")

    nfeatures = graph.attr_matrix.shape[1]
    nclasses = max(labels_all) + 1
    model = model_class(nfeatures, nclasses, **model_args).to(device)

    reg_lambda = torch.tensor(reg_lambda, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloaders = get_dataloaders(idx_all, labels_all)
    early_stopping = EarlyStopping(model, **stopping_args)
    attr_mat_norm_np = normalize_attributes(graph.attr_matrix)
    attr_mat_norm = matrix_to_torch(attr_mat_norm_np).to(device)

    epoch_stats = {'train': {}, 'stopping': {}}

    start_time = time.time()
    last_time = start_time
    for epoch in range(early_stopping.max_epochs):
        for phase in epoch_stats.keys():

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0
            running_corrects = 0

            for idx, labels in dataloaders[phase]:
                idx = idx.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):


                    log_preds = model(attr_mat_norm, idx)
                    preds = torch.argmax(log_preds, dim=1)

                    # Calculate loss
                    cross_entropy_mean = F.nll_loss(log_preds, labels)
                    l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                    loss = cross_entropy_mean + reg_lambda / 2 * l2_reg

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Collect statistics
                    running_loss += loss.item() * idx.size(0)
                    running_corrects += torch.sum(preds == labels)


            # Collect statistics
            epoch_stats[phase]['loss'] = running_loss / len(dataloaders[phase].dataset)
            epoch_stats[phase]['acc'] = running_corrects.item() / len(dataloaders[phase].dataset)

        if epoch % print_interval == 0:
            duration = time.time() - last_time
            last_time = time.time()
            print(f"Epoch{epoch}:  "
                         f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                         f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                         f"early stopping loss = {epoch_stats['stopping']['loss']:.2f}, "
                         f"early stopping acc = {epoch_stats['stopping']['acc'] * 100:.1f} "
                         f"({duration:.3f} sec)")

            logging.info(f"Epoch {epoch}: "
                         f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                         f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                         f"early stopping loss = {epoch_stats['stopping']['loss']:.2f}, "
                         f"early stopping acc = {epoch_stats['stopping']['acc'] * 100:.1f} "
                         f"({duration:.3f} sec)")

        if len(early_stopping.stop_vars) > 0:
            stop_vars = [epoch_stats['stopping'][key]
                         for key in early_stopping.stop_vars]
            if early_stopping.check(stop_vars, epoch):
                break
    runtime = time.time() - start_time
    runtime_perepoch = runtime / (epoch + 1)
    logging.log(22, f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch} ({runtime:.3f} sec)")

    # Load best model weights

    model.load_state_dict(early_stopping.best_state, False)

    train_preds = get_predictions(model, attr_mat_norm, idx_all['train'])
    train_acc = (train_preds == labels_all[idx_all['train']]).mean()

    stopping_preds = get_predictions(model, attr_mat_norm, idx_all['stopping'])
    stopping_acc = (stopping_preds == labels_all[idx_all['stopping']]).mean()
    logging.log(21, f"Early stopping accuracy: {stopping_acc * 100:.1f}%")

    valtest_preds = get_predictions(model, attr_mat_norm, idx_all['valtest'])
    valtest_acc = (valtest_preds == labels_all[idx_all['valtest']]).mean()
    valtest_name = 'Test' if test else 'Validation'
    logging.log(22, f"{valtest_name} accuracy: {valtest_acc * 100:.1f}%")
    
    result = {}
    result['predictions'] = get_predictions(model, attr_mat_norm, torch.arange(len(labels_all)))
    result['train'] = {'accuracy': train_acc}
    result['early_stopping'] = {'accuracy': stopping_acc}
    result['valtest'] = {'accuracy': valtest_acc}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime_perepoch

    return model, result


def get_predictions(model, attr_matrix, idx, batch_size=None):
    if batch_size is None:
        batch_size = idx.numel()
    dataset = TensorDataset(idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    preds = []
    for idx, in dataloader:
        idx = idx.to(attr_matrix.device)
        with torch.set_grad_enabled(False):
            log_preds = model(attr_matrix, idx)
            preds.append(torch.argmax(log_preds, dim=1))
    return torch.cat(preds, dim=0).cpu().numpy()
