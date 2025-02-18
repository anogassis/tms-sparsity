# Define the path to your data directory and version
# data_directory = "../data/"
import pandas as pd
import itertools
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import TensorDataset

# from torch.nn import functional as F
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD
from devinterp.utils import evaluate_mse

from tms.utils.config import DEVICE
from tms.utils.logger import logger
from tms.utils.utils import iterate_container
from tms.data.dataset import SyntheticBinaryValued
from tms.models.autoencoder import ToyAutoencoder


NUM_FEATURES = 6
NUM_HIDDEN_UNITS = 2


def sweep_lambdahat_estimation_hyperparams(
    model,
    dataset,
    weights,
    snapshot_index=-1,
    device=DEVICE,
    sgld_kwargs=None,
    num_draws=100,
    num_chains=100,
    hyperparam_combos=list(
        itertools.product([1, 10, 30, 100, 300], [1e-5, 1e-4, 1e-3, 1e-2])
    ),
    num_burnin_steps=0,
):
    logger.debug(f"Running sweep_lambdahat_estimation_hyperparams")
    observations = []

    sgld_kwargs = sgld_kwargs or {}
    sgld_kwargs.setdefault("localization", 1.0)

    for batch_size, lr in tqdm(
        hyperparam_combos,
        desc="Sweeping hyperparameters",
    ):
        logger.debug(
            f"Running sweep_lambdahat_estimation_hyperparams for batch_size={batch_size}, lr={lr}"
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.load_state_dict(
            {k: torch.Tensor(v) for k, v in weights[snapshot_index].items()}
        )

        logger.debug(f"Model loaded for batch_size={batch_size}, lr={lr}")
        observation = estimate_learning_coeff_with_summary(
            model,
            loader,
            evaluate=evaluate_mse,
            device=device,
            sampling_method=SGLD,
            optimizer_kwargs={"lr": lr, **sgld_kwargs},
            verbose=False,
            num_draws=num_draws,
            num_chains=num_chains,
            online=True,
            num_burnin_steps=num_burnin_steps,
        )
        logger.debug(
            f"Finished estimate for learning coefficient with summary for batch_size={batch_size}, lr={lr}"
        )

        for t_sgld in range(num_draws):
            observations.append(
                {
                    "llc": observation["llc/means"][t_sgld].item(),
                    "llc/std": observation["llc/stds"][t_sgld].item(),
                    "batch_size": batch_size,
                    "lr": lr,
                    "t_sgld": t_sgld,
                    "llc_type": "mean",
                    "loss": observation["loss/trace"][:, t_sgld].mean().item(),
                    "snapshot_index": snapshot_index,
                }
            )

            for llc_type in range(num_chains):
                observations.append(
                    {
                        "llc": observation["llc/trace"][llc_type, t_sgld].item(),
                        "batch_size": batch_size,
                        "lr": lr,
                        "t_sgld": t_sgld,
                        "llc_type": str(llc_type),
                        "loss": observation["loss/trace"][llc_type, t_sgld].item(),
                        "snapshot_index": snapshot_index,
                    }
                )
    logger.debug(f"Returning observations")
    return pd.DataFrame(observations)


def estimate_llc(
    results,
    version,
    data_directory="data",
    hyperparam_combos=[(300, 0.001)],
    snapshot_indices=[0, 9, 18, 27, 36, 45],
    num_samples_test=200,
    num_chains=5,
    num_draws=500,
    num_burnin_steps=0,
):
    llc_estimate_filename = f"{data_directory}/llc_estimate_{version}"
    logger.debug(f"Estimating LLC for version {version}")
    # Load the model
    bias = results[0]["parameters"]["no_bias"]
    num_features = results[0]["parameters"]["m"]
    num_hidden_units = results[0]["parameters"]["n"]
    model = ToyAutoencoder(num_features, num_hidden_units, final_bias=not bias)
    logger.debug(f"Model loaded for version {version}")
    for index in tqdm(range(len(results))):
        for snapshot_index in snapshot_indices:
            file_name = f"{llc_estimate_filename}_{index}_{snapshot_index}_{hyperparam_combos[0][0]}_{hyperparam_combos[0][1]}_{num_chains}_{num_draws}.csv"
            logger.info(f"Running llc estimation for run {index}")
            sparsity = results[index]["parameters"]["sparsity"]

            dataset = SyntheticBinaryValued(num_samples_test, num_features, sparsity)
            dataset_double = TensorDataset(dataset.data, dataset.data)

            logger.info(f"Running llc estimation for snapshot {snapshot_index}")

            llc_estimate = sweep_lambdahat_estimation_hyperparams(
                model,
                dataset_double,
                results[index]["weights"],
                snapshot_index=snapshot_index,
                num_chains=num_chains,
                num_draws=num_draws,
                hyperparam_combos=hyperparam_combos,
                num_burnin_steps=num_burnin_steps,
            )
            logger.debug(f"Saving llc estimate to {file_name}")
            llc_estimate.to_csv(file_name)

    return get_llc_data(results, version, data_directory)


def get_llc_data(results, version, data_directory):
    # List all files in the directory that match the pattern
    dfs = []

    # Initialize an empty list to store DataFrames
    llc_data = pd.DataFrame()
    # Loop through the file paths, read each file and append to the list
    logger.debug(f"Getting llc data for version {version}")
    # TODO: get the index in a more sophisticated way

    for result in iterate_container(results):
        index = result['run_id']
        logger.debug(f"Getting llc data for run {index}")
        pattern = f"llc_estimate_{version}_{index}_"
        logger.debug(f"Pattern: {pattern}")
        file_paths = [
            os.path.join(data_directory, f)
            for f in os.listdir(data_directory)
            if f.endswith(".csv") and pattern in f
        ]
        logger.debug(f"File paths: {file_paths}")
        for file_path in file_paths:

            df = pd.read_csv(file_path)
            df["index"] = index
            dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    logger.debug(f"Concatenating llc data")
    llc_data = pd.concat(dfs, ignore_index=True)
    logger.debug(f"LLC data shape: {llc_data.shape}")
    return llc_data
