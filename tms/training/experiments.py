from tms.utils.logger import logger
import numpy as np
import itertools
import os
import pickle
from typing import Any, Callable, Dict, List
from multiprocessing import Pool, cpu_count
from tms.utils.logger import logger
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm


def _run_single_experiment(args):
    """
    Worker function for a single experiment run.
    Separated out to be picklable for multiprocessing.
    """
    run_id, combination, param_names, train_func, save, file_name = args

    params = dict(zip(param_names, combination))

    if "use_optimal_solution" not in params.keys():
        raise Exception(
            "use_optimal_solution needs to be defined. Update the training dictionary and rerun this function."
        )

    # Calculate `log_ivl` based on `num_epochs`
    num_epochs = params.get("num_epochs", 100)
    num_observations = 50
    steps = sorted(
        list(
            set(np.logspace(0, np.log10(num_epochs), num_observations).astype(int))
        )
    )
    params["log_ivl"] = steps

    pkl_file_name = file_name + "_" + str(run_id) + ".pkl" if (save and file_name) else None

    # Skip if already exists
    if pkl_file_name and os.path.exists(pkl_file_name):
        logger.info(f"Run {run_id} already exists, skipping")
        return run_id, None  # Signal that we skipped this

    logger.info(f"Starting run {run_id}")
    logs, weights, dataset, dataset_test = train_func(**params)

    run_result = {
        "run_id": run_id,
        "parameters": params,
        "logs": logs,
        "weights": weights,
        "dataset": dataset,
        "dataset_test": dataset_test,
    }

    if pkl_file_name:
        with open(pkl_file_name, "wb") as file:
            pickle.dump(run_result, file)
        logger.debug(f"Run {run_id} completed and saved")

    return run_id, run_result



from collections import defaultdict

def run_experiments_batched(
    training_dict: Dict[str, List[Any]],
    train_func: Callable[[Dict[str, Any]], Any],
    save: bool = False,
    file_name: str = None,
    n_jobs: int = None,
) -> List[Dict[str, Any]]:
    """
    Runs experiments with parallelization within sparsity groups.
    Models with the same sparsity share data and train in parallel batches.
    """
    param_names = list(training_dict.keys())
    param_values = [training_dict[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    # Group by sparsity
    sparsity_idx = param_names.index("sparsity")
    seed_idx = param_names.index("seed")

    sparsity_groups = defaultdict(list)
    for run_id, combination in enumerate(combinations):
        sparsity_val = combination[sparsity_idx]
        sparsity_groups[sparsity_val].append((run_id, combination))

    logger.info(f"Running {len(combinations)} experiments across {len(sparsity_groups)} sparsity groups")

    all_results = [None] * len(combinations)

    for sparsity_val, group_runs in sparsity_groups.items():
        logger.info(f"Processing sparsity={sparsity_val} with {len(group_runs)} runs")

        # Extract base params from first run
        first_run_id, first_combination = group_runs[0]
        base_params = dict(zip(param_names, first_combination))

        # Collect all seeds for this sparsity
        all_seeds = [combo[seed_idx] for _, combo in group_runs]

        # Update params for batched training
        base_params['n_parallel_models'] = len(group_runs)
        base_params['all_seeds'] = all_seeds

        # Remove 'seed' from base_params since we're using 'all_seeds' instead
        base_params.pop('seed', None)

        # Train all models for this sparsity in parallel - unpack the dict
        logs_list, weights_list, dataset, dataset_test = train_func(**base_params)

        # Unpack results back to individual runs
        for idx, (run_id, combination) in enumerate(group_runs):
            params = dict(zip(param_names, combination))
            result = {
                "run_id": run_id,
                "params": params,
                "logs": logs_list[idx],
                "weights": weights_list[idx],
                "dataset": dataset,
                "dataset_test": dataset_test,
            }

            all_results[run_id] = result

            # Save individual result
            if save:
                pkl_file_name = f"{file_name}_{run_id}.pkl"
                with open(pkl_file_name, "wb") as file:
                    pickle.dump(result, file)

    logger.info("All runs completed")

    if save:
        with open(f"{file_name}_all_runs.pkl", "wb") as file:
            pickle.dump(all_results, file)
        logger.info(f"All results saved to {file_name}_all_runs.pkl")

    return all_results
