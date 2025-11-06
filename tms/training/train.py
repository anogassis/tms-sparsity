import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Iterable, Optional, Union, Tuple, List, Dict, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tms.utils.config import DEVICE
from tms.data.dataset import SyntheticBinaryValued, SyntheticDataset
from tms.models.autoencoder import ToyAutoencoder, BatchedToyAutoencoder
from tms.utils.utils import generate_init_param, generate_optimal_solution
import pickle



def create_and_train(
    m: int,
    n: int,
    num_samples: int,
    num_samples_test: int,
    batch_size: Optional[int] = 1,
    num_epochs: int = 100,
    sparsity: Union[float, int] = 1,
    lr: float = 0.001,
    log_ivl: Iterable[int] = [],
    device=DEVICE,
    momentum=0.9,
    weight_decay=0.0,
    init_kgon: int = None,
    no_bias: bool = False,
    init_zerobias: bool = False,
    prior_std: float = 10.0,
    seed: int = 0,
    use_optimal_solution: bool = False,
    data_generating_class: SyntheticDataset = SyntheticBinaryValued,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Create and train a model using the given parameters.

    Parameters
    ----------
    m : int
        The number of input features.
    n : int
        The number of output features.
    num_samples : int
        The number of training samples.
    num_samples_test : int
        The number of testing samples.
    batch_size : Optional[int], optional
        The batch size for training, by default 1.
    num_epochs : int, optional
        The number of training epochs, by default 100.
    sparsity : Union[float, int], optional
        The sparsity level of the training data, by default 1.
    lr : float, optional
        The learning rate for the optimizer, by default 0.001.
    log_ivl : Iterable[int], optional
        The intervals at which to log the training progress, by default [].
    device : _type_, optional
        The device to use for training, by default DEVICE.
    momentum : float, optional
        The momentum factor for the optimizer, by default 0.9.
    weight_decay : float, optional
        The weight decay factor for the optimizer, by default 0.0.
    init_kgon : _type_, optional
        The initialization method for the model weights, by default None.
    no_bias : bool, optional
        Whether to exclude bias terms in the model, by default False.
    init_zerobias : bool, optional
        Whether to initialize bias terms to zero, by default False.
    prior_std : _type_, optional
        The standard deviation of the prior distribution for weight initialization, by default 10.
    seed : int, optional
        The random seed for reproducibility, by default 0.
    use_optimal_solution : bool, optional
        Whether to use an optimal solution for weight initialization, by default False.

    Returns
    -------
    logs : pandas.DataFrame
        A DataFrame containing the training logs.
    weights : list
        A list of dictionaries containing the model weights at different training steps.
    """
    torch.manual_seed(seed)

    model = ToyAutoencoder(m, n, final_bias=True)
    init_weights = generate_init_param(
        n,
        m,
        init_kgon,
        prior_std=prior_std,
        no_bias=no_bias,
        init_zerobias=init_zerobias,
        seed=seed,
    )

    if use_optimal_solution:
        init_weights = generate_optimal_solution(n, m, rot=0.0)

    if "b" in init_weights:
        model.unembedding.bias.data = torch.from_numpy(
            init_weights["b"].flatten()
        ).float()

    model.embedding.weight.data = torch.from_numpy(init_weights["W"]).float()

    dataset = data_generating_class(num_samples, m, sparsity)

    dataset_test = data_generating_class(num_samples_test, m, sparsity)
    batch_size = batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    logs = pd.DataFrame(
        [
            {
                "loss": None,
                "acc": None,
                "test_loss": None,
                "test_acc": None,
                "step": step,
            }
            for step in log_ivl
        ]
    )

    model.to(device)
    weights = []

    def log(step):
        loss = 0.0
        loss_test = 0.0
        acc = 0.0
        acc_test = 0.0
        length = 0
        length_test = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                outputs = model(batch)
                loss += criterion(outputs, batch).item() * len(
                    batch
                )  # adding "* len(batch)"
                acc += (outputs.round() == batch).float().sum().item()
                length += len(batch)
            for batch in dataloader_test:
                batch = batch.to(device)
                outputs = model(batch)
                loss_test += criterion(outputs, batch).item() * len(batch)
                acc_test += (outputs.round() == batch).float().sum().item()
                length_test += len(batch)

        loss /= length
        acc /= length
        loss_test /= length_test
        acc_test /= length_test

        logs.loc[logs["step"] == step, ["loss", "acc", "test_loss", "test_acc"]] = [
            loss,
            acc,
            loss_test,
            acc_test,
        ]
        weights.append(
            {k: v.cpu().detach().clone().numpy() for k, v in model.state_dict().items()}
        )

    step = 0
    log(step)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        for batch in dataloader:
            batch = batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            step += 1

            if step in log_ivl:
                log(step)

    return logs, weights, dataset, dataset_test


def create_and_train_batched(
    m: int,
    n: int,
    num_samples: int,
    num_samples_test: int,
    batch_size: int = 1024,
    num_epochs: int = 100,
    sparsity: float = 1.0,
    lr: float = 0.001,
    log_ivl: List[int] = None,
    device: str = "cuda",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    init_kgon: int = None,
    no_bias: bool = False,
    init_zerobias: bool = False,
    prior_std: float = 10.0,
    seed: int = 0,
    use_optimal_solution: bool = False,
    data_generating_class=None,
    n_parallel_models: int = 1,
    all_seeds: List[int] = None,
) -> Tuple[List[pd.DataFrame], List[List[Dict[str, Any]]], Any, Any]:
    """
    Train multiple models in parallel on the same data.
    """
    if log_ivl is None:
        log_ivl = []

    if all_seeds is None:
        all_seeds = [seed + i for i in range(n_parallel_models)]

    assert len(all_seeds) == n_parallel_models, "all_seeds length must match n_parallel_models"

    # Create batched model
    model = BatchedToyAutoencoder(m, n, n_parallel_models, final_bias=True)

    # Initialize weights for each model separately
    for model_idx, model_seed in enumerate(all_seeds):
        torch.manual_seed(model_seed)
        init_weights = generate_init_param(
            n, m, init_kgon,
            prior_std=prior_std,
            no_bias=no_bias,
            init_zerobias=init_zerobias,
            seed=model_seed,
        )

        if use_optimal_solution:
            init_weights = generate_optimal_solution(n, m, rot=0.0)

        # Set weights for this specific model in the batch
        model.embedding_weight.data[model_idx] = torch.from_numpy(init_weights["W"]).float()

        if "b" in init_weights and model.unembedding_bias is not None:
            model.unembedding_bias.data[model_idx] = torch.from_numpy(
                init_weights["b"].flatten()
            ).float()

    # Create shared datasets
    torch.manual_seed(seed)
    dataset = data_generating_class(num_samples, m, sparsity)
    dataset_test = data_generating_class(num_samples_test, m, sparsity)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='none')

    # Prepare logs for each model - FIXED: ensure columns exist even if log_ivl is empty
    logs_list = [
        pd.DataFrame({
            "loss": [None] * len(log_ivl),
            "acc": [None] * len(log_ivl),
            "test_loss": [None] * len(log_ivl),
            "test_acc": [None] * len(log_ivl),
            "step": log_ivl,
        })
        for _ in range(n_parallel_models)
    ]

    weights_list = [[] for _ in range(n_parallel_models)]

    model.to(device)

    def log(step):
        """Log metrics for all models."""
        losses = torch.zeros(n_parallel_models, device=device)
        accs = torch.zeros(n_parallel_models, device=device)
        losses_test = torch.zeros(n_parallel_models, device=device)
        accs_test = torch.zeros(n_parallel_models, device=device)

        with torch.no_grad():
            # Training set metrics
            total_samples = 0
            for batch in dataloader:
                batch = batch.to(device)
                outputs = model(batch)

                batch_expanded = batch.unsqueeze(0).expand(n_parallel_models, -1, -1)

                batch_losses = criterion(outputs, batch_expanded).mean(dim=[1, 2])
                losses += batch_losses * len(batch)

                batch_accs = (outputs.round() == batch_expanded).float().mean(dim=[1, 2])
                accs += batch_accs * len(batch)

                total_samples += len(batch)

            losses /= total_samples
            accs /= total_samples

            # Test set metrics
            total_samples_test = 0
            for batch in dataloader_test:
                batch = batch.to(device)
                outputs = model(batch)
                batch_expanded = batch.unsqueeze(0).expand(n_parallel_models, -1, -1)

                batch_losses = criterion(outputs, batch_expanded).mean(dim=[1, 2])
                losses_test += batch_losses * len(batch)

                batch_accs = (outputs.round() == batch_expanded).float().mean(dim=[1, 2])
                accs_test += batch_accs * len(batch)

                total_samples_test += len(batch)

            losses_test /= total_samples_test
            accs_test /= total_samples_test

        # Update logs for each model
        for model_idx in range(n_parallel_models):
            # FIXED: Use .loc properly
            mask = logs_list[model_idx]["step"] == step
            logs_list[model_idx].loc[mask, "loss"] = losses[model_idx].item()
            logs_list[model_idx].loc[mask, "acc"] = accs[model_idx].item()
            logs_list[model_idx].loc[mask, "test_loss"] = losses_test[model_idx].item()
            logs_list[model_idx].loc[mask, "test_acc"] = accs_test[model_idx].item()

            # Store weights for this model
            model_state = {
                "embedding.weight": model.embedding_weight[model_idx].cpu().detach().clone().numpy(),
            }
            if model.unembedding_bias is not None:
                model_state["unembedding.bias"] = model.unembedding_bias[model_idx].cpu().detach().clone().numpy()

            weights_list[model_idx].append(model_state)

    # Initial logging
    step = 0
    log(step)

    # Training loop
    for epoch in tqdm(range(num_epochs), desc=f"Training {n_parallel_models} models"):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            batch_expanded = batch.unsqueeze(0).expand(n_parallel_models, -1, -1)

            loss = criterion(outputs, batch_expanded).mean()

            loss.backward()
            optimizer.step()

            step += 1
