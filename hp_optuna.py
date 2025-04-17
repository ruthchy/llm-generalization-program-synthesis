import yaml
import json
import os
import sys
import subprocess
import time
import wandb
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("hp_optuna.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_hyperparameter_space(file_path="hyperparameter_grid.yaml"):
    """Load hyperparameter search space definition."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def create_config_for_run(base_config_path, hp_params, output_path="config_temp.yaml"):
    """Create a temporary config file with current hyperparameters."""
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with hyperparameters
    config['training']['learning_rate'] = hp_params['learning_rate']
    config['training']['per_device_train_batch_size'] = hp_params['per_device_train_batch_size']
    config['lora']['rank'] = hp_params['lora_rank']
    config['lora']['alpha'] = hp_params['lora_alpha']
    
    # Test mode: reduce epochs
    if TEST_MODE:
        config['training']['train_epochs'] = 1
        
    # Save updated config
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    return output_path

def extract_wandb_run_id(log_content):
    """Extract the WandB run ID from pipeline log output"""
    # Look for wandb run URL pattern
    match = re.search(r'wandb: Run data is saved locally in .*?run-\d+_\d+-([a-z0-9]+)', log_content)
    if match:
        return match.group(1)
    
    # Alternative pattern
    match = re.search(r'wandb: View run at (.*?)$', log_content, re.MULTILINE)
    if match:
        url = match.group(1).strip()
        # Extract run ID from URL
        url_parts = url.split('/')
        if len(url_parts) > 0:
            return url_parts[-1]
    
    return None

def get_wandb_metrics(run_id, wait_timeout=120):
    """Fetch metrics from a WandB run"""
    try:
        api = wandb.Api()
        entity = "priscillachyrva-university-mannheim"  # Your WandB username or organization
        project = "master-thesis"  # Your project name
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Wait for metrics to be available (up to timeout)
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            summary = run.summary._json_dict
            if 'eval/comp_loss' in summary:
                logger.info(f"Successfully retrieved metrics for run {run_id}")
                return summary
            
            # Wait and refresh
            logger.info(f"Waiting for metrics to sync for run {run_id}... ({int(time.time() - start_time)}s)")
            time.sleep(10)  # Check every 10 seconds
            run = api.run(f"{entity}/{project}/{run_id}")
            
        logger.warning(f"Timed out waiting for metrics for run {run_id}")
        return run.summary._json_dict  # Return whatever we have
    except Exception as e:
        logger.error(f"Error fetching WandB metrics: {e}")
        return None

def run_pipeline(config_path, trial_id):
    """Run pipeline with a specific config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"hp_optuna_trial_{trial_id}_{timestamp}.log"
    log_content = ""
    
    logger.info(f"Starting trial {trial_id} at {timestamp}")
    logger.info(f"Logs will be saved to {log_file}")
    
    try:
        if TEST_MODE:
            wb_type = "test_optimize"
            fraction = "0.1"
        else:
            fraction = "1.0"
            wb_type = "optimize"

        # Use your existing pipeline.py script
        cmd = [
            "python", "pipeline.py",
            "--fine_tune",
            "--sample_fraction", fraction,
            "--config", config_path,
            "--wb_type", wb_type
        ]
        
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Track if OOM error was detected
            oom_detected = False
            cuda_error_detected = False
            metrics = {}
            wandb_run_id = None
            
            for line in process.stdout:
                print(line, end='')  # Print to console
                log_f.write(line)    # Write to log file
                log_content += line  # Store for later analysis
                log_f.flush()        # Make sure it's written immediately
                
                # Try to extract wandb run ID as early as possible
                if not wandb_run_id and "wandb:" in line:
                    possible_id = extract_wandb_run_id(line)
                    if possible_id:
                        wandb_run_id = possible_id
                        logger.info(f"Detected WandB run ID early: {wandb_run_id}")
                
                # Extract evaluation metric if available
                if "eval/comp_loss" in line:
                    try:
                        metrics['comp_loss'] = float(line.split("eval/comp_loss:")[1].split()[0])
                    except:
                        pass
                        
                # Check for common error patterns
                if "CUDA out of memory" in line:
                    oom_detected = True
                if "CUDA error" in line:
                    cuda_error_detected = True
            
            process.wait()
            
            # If wandb_run_id wasn't found earlier, try to extract it from the full log
            if not wandb_run_id:
                wandb_run_id = extract_wandb_run_id(log_content)
                if wandb_run_id:
                    logger.info(f"Extracted WandB run ID from full log: {wandb_run_id}")
            
            # Check exit status and error indicators
            if process.returncode != 0 or oom_detected or cuda_error_detected:
                error_type = "Unknown error"
                if oom_detected:
                    error_type = "CUDA out of memory"
                elif cuda_error_detected:
                    error_type = "CUDA error"
                
                logger.error(f"Trial {trial_id} failed with {error_type} (exit code {process.returncode})")
                log_f.write(f"\n\nTrial failed with {error_type} (exit code {process.returncode})")
                return None, error_type
            
            # Try to get metrics from WandB if run ID found
            if wandb_run_id:
                wandb_metrics = get_wandb_metrics(wandb_run_id)
                if wandb_metrics and 'eval/comp_loss' in wandb_metrics:
                    metrics['comp_loss'] = wandb_metrics['eval/comp_loss']
                    logger.info(f"Using WandB metrics: {metrics['comp_loss']}")
            
            # Return metrics if found, otherwise use a placeholder
            if metrics:
                return metrics, None
            else:
                logger.warning(f"No metrics found in trial {trial_id} output")
                return {"comp_loss": 999.0}, "No metrics found"  # Placeholder for failed extraction
    except Exception as e:
        logger.exception(f"Error running pipeline: {e}")
        return None, str(e)

class OptunaStorage:
    """Simple storage for Optuna study to persist between SLURM jobs"""
    def __init__(self, file_path="optuna_storage.json"):
        self.file_path = file_path
        self.load()
        
    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "trials": [],
                "best_trial": None,
                "best_value": float('inf'),
                "oom_failures": []  # Track OOM failures
            }
    
    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_trial(self, trial_id, params, value, error=None):
        trial = {
            "trial_id": trial_id,
            "params": params,
            "value": value,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if error:
            trial["error"] = error
            # Track OOM failures for memory-aware ordering
            if "CUDA out of memory" in error:
                self.data.setdefault("oom_failures", []).append({
                    "params": params,
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        self.data["trials"].append(trial)
        
        # Update best trial if this is better
        if value is not None and value < self.data["best_value"]:
            self.data["best_value"] = value
            self.data["best_trial"] = trial
        
        self.save()
    
    def get_next_trial(self):
        """Get next trial sorted by memory footprint (lowest first)"""
        if not self.pending_trials:
            return None
            
        # Sort by memory footprint (ascending)
        self.pending_trials.sort(key=lambda x: x[1])
        return self.pending_trials.pop(0)[0]

def objective(trial):
    """Optuna objective function with OOM pruning."""
    # Load hyperparameter space
    hp_space = load_hyperparameter_space()["hyperparameter_grid"]

    # Suggest hyperparameters
    params = {
        "learning_rate": trial.suggest_categorical("learning_rate", hp_space["learning_rate"]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", hp_space["per_device_train_batch_size"]),
        "lora_rank": trial.suggest_categorical("lora_rank", hp_space["lora_rank"]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", hp_space["lora_alpha"]),
    }

    # Create config file
    config_path = create_config_for_run("config.yaml", params)

    # Run the pipeline
    metrics, error = run_pipeline(config_path, trial.number)

    # Save to persistent storage
    storage = OptunaStorage()

    # Save to persistent storage
    if metrics is None:
        # Trial failed completely, return a penalty value
        logger.warning(f"Trial {trial.number} failed with error: {error}")
        storage.add_trial(trial.number, params, 999.0, error)
        return 999.0  # Large penalty value
    else:
        # Extract the primary metric (completion loss)
        comp_loss = metrics.get("comp_loss", 999.0)
        storage.add_trial(trial.number, params, comp_loss, error if error else None)
        return comp_loss

def run_optuna_optimization(n_trials, timeout):
    """Run Optuna hyperparameter optimization"""
    # Create study or load existing one
    storage_file = "optuna_test_storage.json" if TEST_MODE else "optuna_storage.json"
    study_name = "llm_finetuning_test" if TEST_MODE else "llm_finetuning_optimization"
    
    # Check if we have an existing study to resume
    if os.path.exists(storage_file):
        logger.info("Resuming from existing Optuna study")
    else:
        logger.info("Creating new Optuna study")
    
    # Create a study with the default TPE sampler
    study = optuna.create_study(
        study_name=study_name, 
        direction="minimize",  # Minimize the loss
        sampler=TPESampler(seed=42),
        load_if_exists=True
    )
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except Exception as e:
        logger.exception(f"Optimization stopped: {e}")
    finally:
        # Report best result
        if study.best_trial:
            logger.info("Best trial:")
            logger.info(f"  Value: {study.best_trial.value}")
            logger.info("  Params:")
            for key, value in study.best_trial.params.items():
                logger.info(f"    {key}: {value}")
            
            # Save best config
            best_params = study.best_trial.params
            create_config_for_run("config.yaml", best_params, "best_config.yaml")
            logger.info("Best configuration saved to best_config.yaml")
        else:
            logger.warning("No best trial found.")
    
    # Create completion marker if we've done all trials
    storage = OptunaStorage()
    if len(storage.data["trials"]) >= n_trials:
        with open("hp_tuning_completed.marker", "w") as f:
            f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("All trials completed. Created completion marker.")
    else:
        logger.info(f"Completed {len(storage.data['trials'])}/{n_trials} trials.")
    
    return storage.data["best_trial"] if "best_trial" in storage.data else None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning.")
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode (default: False).")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of trials to run.")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds.") # Optional
    args = parser.parse_args()

    TEST_MODE = args.test_mode

    # Run optimization
    best_trial = run_optuna_optimization(n_trials=args.n_trials , timeout=args.timeout)
    
    # Generate report
    if best_trial:
        print(f"\nBest hyperparameters found (loss = {best_trial['value']}):")
        for param, value in best_trial['params'].items():
            print(f"{param}: {value}")