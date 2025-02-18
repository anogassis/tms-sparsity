import numpy as np
import itertools
import os
import pickle
from typing import Any, Callable, Dict, List

from tms.utils.logger import logger


def run_experiments(
    training_dict: Dict[str, List[Any]],
    train_func: Callable[[Dict[str, Any]], Any],
    save: bool = False,
    file_name: str = None,
) -> List[Dict[str, Any]]:
    """
    Runs experiments for all combinations of parameters in the training dictionary,
    with an incremental run_id starting at 0.

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

    # Iterate through each combination
    for run_id, combination in enumerate(combinations):

        params = dict(zip(param_names, combination))

        if "use_optimal_solution" not in params.keys():
            raise Exception(
                "use_optimal_solution needs to be defined. Update the training dictionary and rerun this function."
            )

        # Calculate `log_ivl` based on `num_epochs`
        num_epochs = params.get("num_epochs", 100)
        num_observations = 50  # As per example
        steps = sorted(
            list(
                set(np.logspace(0, np.log10(num_epochs), num_observations).astype(int))
            )
        )
        params["log_ivl"] = steps
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

        if save and file_name:
            pkl_file_name = file_name + "_" + str(run_id) + ".pkl"
            logger.debug(f"Saving results to {pkl_file_name}")
        if os.path.exists(pkl_file_name):
            continue
        else:
            with open(pkl_file_name, "wb") as file:
                pickle.dump(run_result, file)
                logger.debug(f"Run {run_id} completed")

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
