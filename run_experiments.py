import os
import tms.training.experiments as experiments
import tms.utils.utils as utils
import tms.utils.config as config
import tms.training.train as train
from tms.utils.utils import load_results
from tms.llc import estimate_llc, get_llc_data
from tms.utils.logger import logger
import multiprocessing as mp


mp.set_start_method('spawn', force=True)

def run_all_experiments(versions, data_path, parallel_experiments=16):
    """
    Train multiple models in parallel on the same GPU.
    """
    num_models = len(models)
    criterion = nn.MSELoss()

    # Initialize logs and weights for each model
    all_logs = []
    all_weights = []
    datasets = []
    datasets_test = []

    for idx, params in enumerate(params_list):
        log_ivl = params["log_ivl"]
        logs = pd.DataFrame([
            {
                "loss": None,
                "acc": None,
                "test_loss": None,
                "test_acc": None,
                "step": step,
            }
            for step in log_ivl
        ])
        all_logs.append(logs)
        all_weights.append([])
        datasets.append(dataloaders[idx].dataset)
        datasets_test.append(dataloaders_test[idx].dataset)

    def log_model(model_idx, step):
        """Log metrics for a single model."""
        model = models[model_idx]
        dataloader = dataloaders[model_idx]
        dataloader_test = dataloaders_test[model_idx]
        logs = all_logs[model_idx]

        loss_total = 0.0
        loss_test_total = 0.0
        acc_total = 0.0
        acc_test_total = 0.0
        length = 0
        length_test = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                outputs = model(batch)
                loss_total += criterion(outputs, batch).item() * len(batch)
                acc_total += (outputs.round() == batch).float().sum().item()
                length += len(batch)

            for batch in dataloader_test:
                batch = batch.to(device)
                outputs = model(batch)
                loss_test_total += criterion(outputs, batch).item() * len(batch)
                acc_test_total += (outputs.round() == batch).float().sum().item()
                length_test += len(batch)

        loss = loss_total / length
        acc = acc_total / length
        loss_test = loss_test_total / length_test
        acc_test = acc_test_total / length_test

        logs.loc[logs["step"] == step, ["loss", "acc", "test_loss", "test_acc"]] = [
            loss, acc, loss_test, acc_test
        ]

        all_weights[model_idx].append({
            k: v.cpu().detach().clone().numpy()
            for k, v in model.state_dict().items()
        })

    # Log initial state for all models
    for model_idx in range(num_models):
        log_model(model_idx, step=0)

    # Get max epochs across all models
    max_epochs = max(params["num_epochs"] for params in params_list)

    # Create iterators for each dataloader
    data_iters = [iter(dataloader) for dataloader in dataloaders]
    steps = [0] * num_models

    # Training loop - interleave training steps across models
    with tqdm(total=max_epochs, desc="Training (batched)") as pbar:
        for epoch in range(max_epochs):
            for model_idx in range(num_models):
                # Skip if this model has finished training
                if epoch >= params_list[model_idx]["num_epochs"]:
                    continue

                model = models[model_idx]
                optimizer = optimizers[model_idx]
                data_iter = data_iters[model_idx]
                log_ivl = params_list[model_idx]["log_ivl"]

                try:
                    # Train one epoch for this model
                    for batch in data_iter:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch)
                        loss = criterion(outputs, batch)
                        loss.backward()
                        optimizer.step()
                        steps[model_idx] += 1

                        # Log if at logging step
                        if steps[model_idx] in log_ivl:
                            log_model(model_idx, steps[model_idx])

                    # Reset iterator for next epoch
                    data_iters[model_idx] = iter(dataloaders[model_idx])

                except StopIteration:
                    # Shouldn't happen with proper epoch counting, but just in case
                    data_iters[model_idx] = iter(dataloaders[model_idx])

            pbar.update(1)

    # Return results in the expected format
    results = []
    for model_idx in range(num_models):
        results.append({
            "logs": all_logs[model_idx],
            "weights": all_weights[model_idx],
            "dataset": datasets[model_idx],
            "dataset_test": datasets_test[model_idx],
        })

    return results


def run_experiments_batched(
    training_dict: Dict[str, List[Any]],
    save: bool = False,
    file_name: str = None,
    parallel_experiments: int = 16,
    device=DEVICE,
) -> List[Dict[str, Any]]:
    """
    Run experiments in parallel batches on a single GPU.

    Args:
        training_dict: Dictionary of parameter lists
        save: Whether to save results
        file_name: Base filename for saving
        parallel_experiments: Number of experiments to run simultaneously
        device: Device to use for training
    """
    param_names = list(training_dict.keys())
    param_values = [training_dict[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    all_results = []

    for chunk_start in range(0, len(combinations), parallel_experiments):
        chunk_end = min(chunk_start + parallel_experiments, len(combinations))

        logger.info(f"Training runs {chunk_start}-{chunk_end-1} in parallel ({chunk_end-chunk_start} models)")

        models = []
        optimizers = []
        dataloaders = []
        dataloaders_test = []
        params_list = []
        active_run_ids = []

        for run_id in range(chunk_start, chunk_end):
            combination = combinations[run_id]
            params = dict(zip(param_names, combination))

            if "use_optimal_solution" not in params:
                raise Exception("use_optimal_solution needs to be defined.")

            # Calculate log_ivl
            num_epochs = params.get("num_epochs", 100)
            num_observations = 50
            steps = sorted(list(set(
                np.logspace(0, np.log10(num_epochs), num_observations).astype(int)
            )))
            params["log_ivl"] = steps

            pkl_file_name = f"{file_name}_{run_id}.pkl" if file_name else None

            # Check if already exists
            if pkl_file_name and os.path.exists(pkl_file_name):
                logger.info(f"Loading cached run {run_id}")
                with open(pkl_file_name, "rb") as file:
                    all_results.append(pickle.load(file))
                continue

            # Setup this experiment
            model, optimizer, dataloader, dataloader_test, dataset, dataset_test = \
                setup_single_experiment(**{k: v for k, v in params.items() if k != "log_ivl"}, device=device)

            models.append(model)
            optimizers.append(optimizer)
            dataloaders.append(dataloader)
            dataloaders_test.append(dataloader_test)
            params_list.append(params)
            active_run_ids.append(run_id)

        # Train all models in this chunk together
        if models:
            chunk_results = train_multiple_models_parallel(
                models, optimizers, dataloaders, dataloaders_test,
                params_list, device=device
            )

            # Save and collect results
            for idx, run_id in enumerate(active_run_ids):
                result = {
                    "run_id": run_id,
                    "parameters": params_list[idx],
                    **chunk_results[idx]
                }
                all_results.append(result)

                if save:
                    pkl_file_name = f"{file_name}_{run_id}.pkl"
                    with open(pkl_file_name, "wb") as file:
                        pickle.dump(result, file)
                    logger.debug(f"Run {run_id} saved")

        # Clean up to free GPU memory
        del models, optimizers, dataloaders, dataloaders_test
        torch.cuda.empty_cache()

    logger.info("All runs completed")

    if save and file_name:
        with open(f"{file_name}_all_runs.pkl", "wb") as file:
            pickle.dump(all_results, file)

    return all_results


def run_all_experiments(versions, data_path, parallel_experiments=16):
    """
    Run experiments for different versions and estimate LLC.
    """
    logger.info("--------------------")
    logger.info(f"Starting experiments for versions={versions}")
    parameters = [config.training_dicts[version] for version in versions]
    file_names = [
        os.path.join(data_path, f"logs_loss_{version}") for version in versions
    ]

    for version, params, file_name in zip(versions, parameters, file_names):
        if os.path.exists(f"{file_name}_all_runs.pkl"):
            logger.info(f"File {file_name}_all_runs.pkl already exists. Skipping.")
            continue

        logger.info(
            f"Running BATCHED experiments for version={version}, parallel={parallel_experiments}"
        )

        results = run_experiments_batched(  # ← Changed function
            params,
            save=True,
            file_name=file_name,
            parallel_experiments=parallel_experiments  # ← New parameter
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
    VERSIONS = ["1.15.0"]
    DATA_PATH = "data"
    run_all_experiments(VERSIONS, DATA_PATH)
