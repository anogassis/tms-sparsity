from tms.utils.logger import logger
import numpy as np
import itertools
import os
import pickle
from typing import Any, Callable, Dict, List
from multiprocessing import Pool, cpu_count
from tms.utils.logger import logger


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


def run_experiments(
    training_dict: Dict[str, List[Any]],
    train_func: Callable[[Dict[str, Any]], Any],
    save: bool = False,
    file_name: str = None,
    n_jobs: int = None,
) -> List[Dict[str, Any]]:
    """
    Runs experiments for all combinations of parameters in the training dictionary,
    with an incremental run_id starting at 0, using multiprocessing.

    Parameters
    ----------
    training_dict : dict
        A dictionary where keys are parameter names and values are lists of parameter values.
    train_func : callable
        A function that takes a dictionary of parameters and returns the result of the training.
    save : bool
        A flag to save the results of each run.
    file_name : str
        The name of the file to save the results.
    n_jobs : int, optional
        Number of parallel jobs. Defaults to cpu_count() - 1.

    Returns
    -------
    List[dict]
        A list of dictionaries, each containing the run_id, parameters used, and the result of the training.
    """
    # Extract parameter names and their values
    param_names = list(training_dict.keys())
    param_values = [training_dict[name] for name in param_names]

    # Generate all combinations of parameters
    combinations = list(itertools.product(*param_values))

    # Prepare arguments for each run
    args_list = [
        (run_id, combination, param_names, train_func, save, file_name)
        for run_id, combination in enumerate(combinations)
    ]

    # Determine number of processes
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Leave one core free

    logger.info(f"Running {len(combinations)} experiments using {n_jobs} processes")

    # Run experiments in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_run_single_experiment, args_list)

    # Collect all results
    all_results = []
    for idx in range(len(combinations)):
        pkl_file_name = file_name + "_" + str(idx) + ".pkl"
        with open(pkl_file_name, "rb") as file:
            all_results.append(pickle.load(file))

    logger.info("All runs completed")

    if save:
        with open(file_name + "_all_runs.pkl", "wb") as file:
            pickle.dump(all_results, file)
        logger.info(f"All results saved to {file_name}_all_runs.pkl")

    return all_results
