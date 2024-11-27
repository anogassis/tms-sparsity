import os
import tms.training.experiments as experiments
import tms.utils.utils as utils
import tms.utils.config as config
import tms.training.train as train
from tms.utils.utils import load_results
from tms.llc import estimate_llc, get_llc_data
from tms.utils.logger import logger


def run_all_experiments(versions, data_path):
    """
    Run experiments for different versions and estimate LLC.

    Args:
        versions (list): List of version strings.
        data_path (str): Path to the data directory.
    """
    logger.info("--------------------")
    logger.info(f"Starting experiments for versions={versions}")

    parameters = [config.training_dicts[version] for version in versions]
    file_names = [
        os.path.join(data_path, f"logs_loss_{version}") for version in versions
    ]

    for version, params, file_name in zip(versions, parameters, file_names):

        # If file exists, skip the experiments
        if os.path.exists(f"{file_name}_all_runs.pkl"):
            logger.info(f"File {file_name}_all_runs.pkl already exists. Skipping.")
            continue

        logger.info(
            f"Running experiments for version={version}, params={params}, file_name={file_name}"
        )
        results = experiments.run_experiments(
            params, train.create_and_train, save=True, file_name=file_name
        )
        logger.info(f"Experiments completed for version={version}")

    for version in versions:
        logger.info(f"Estimating LLC for version={version}")
        # If the last LLC file exists that matches llc_estimate_{version}_449_45*.csv, skip the LLC estimation
        if os.path.exists(f"{data_path}/llc_estimate_{version}_449_45*.csv"):
            logger.info(f"LLC estimates already exist for version={version}. Skipping.")
            continue

        results = load_results(data_path, version)
        logger.debug(f"Results loaded for version={version}")
        llc_estimates = estimate_llc(results, version)
        logger.info(f"LLC estimates for version={version}: {llc_estimates}")


if __name__ == "__main__":
    VERSIONS = ["debug_1.14.0"]
    DATA_PATH = "data"
    run_all_experiments(VERSIONS, DATA_PATH)
