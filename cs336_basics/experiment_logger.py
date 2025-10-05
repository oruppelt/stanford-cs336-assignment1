import json
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np


class ExperimentLogger:
    """
    Logs training metrics with both iteration count and wall-clock time tracking.
    Saves metrics to JSON for post-hoc analysis and visualization.
    """

    def __init__(self, experiment_name: str, log_dir: str = "./logs",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Unique name for this experiment
            log_dir: Directory to save log files
            config: Training configuration/hyperparameters to save
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics = {
            'iterations': [],
            'wall_time': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'tokens_per_sec': []
        }

        # Store configuration
        self.config = config or {}

        # Timing
        self.start_time = time.time()
        self.last_save_time = self.start_time

        # Running averages for smoothing
        self.train_loss_buffer = []
        self.buffer_size = 10

        print(f"ExperimentLogger initialized: {experiment_name}")
        print(f"Log directory: {self.log_dir}")

    def log_step(self, iteration: int, train_loss: Optional[float] = None,
                 val_loss: Optional[float] = None, learning_rate: Optional[float] = None,
                 grad_norm: Optional[float] = None, tokens_per_sec: Optional[float] = None):
        """
        Log metrics for a single training step.

        Args:
            iteration: Current training iteration
            train_loss: Training loss value
            val_loss: Validation loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm after clipping
            tokens_per_sec: Processing speed
        """
        current_time = time.time() - self.start_time

        self.metrics['iterations'].append(iteration)
        self.metrics['wall_time'].append(current_time)

        # Log training loss with smoothing
        if train_loss is not None:
            self.train_loss_buffer.append(train_loss)
            if len(self.train_loss_buffer) > self.buffer_size:
                self.train_loss_buffer.pop(0)
            smoothed_loss = np.mean(self.train_loss_buffer)
            self.metrics['train_loss'].append(smoothed_loss)
        else:
            self.metrics['train_loss'].append(None)

        # Log other metrics
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['tokens_per_sec'].append(tokens_per_sec)

    def save(self, force: bool = False):
        """
        Save metrics to disk. Auto-saves every 5 minutes unless forced.

        Args:
            force: If True, save immediately regardless of time since last save
        """
        current_time = time.time()

        # Auto-save every 5 minutes or on force
        if not force and (current_time - self.last_save_time) < 300:
            return

        log_file = self.log_dir / f"{self.experiment_name}.json"

        # Prepare data for saving
        save_data = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'start_time': self.start_time,
            'metrics': self.metrics
        }

        # Save to JSON
        with open(log_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        self.last_save_time = current_time
        print(f"Metrics saved to {log_file}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the experiment."""
        train_losses = [x for x in self.metrics['train_loss'] if x is not None]
        val_losses = [x for x in self.metrics['val_loss'] if x is not None]

        summary = {
            'experiment_name': self.experiment_name,
            'total_iterations': len(self.metrics['iterations']),
            'total_time': self.metrics['wall_time'][-1] if self.metrics['wall_time'] else 0,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'best_train_loss': min(train_losses) if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
        }
        return summary

    def print_summary(self):
        """Print experiment summary to console."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print(f"Experiment Summary: {summary['experiment_name']}")
        print("=" * 60)
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Total time: {summary['total_time'] / 3600:.2f} hours")
        if summary['final_train_loss']:
            print(f"Final train loss: {summary['final_train_loss']:.4f}")
            print(f"Best train loss: {summary['best_train_loss']:.4f}")
        if summary['final_val_loss']:
            print(f"Final val loss: {summary['final_val_loss']:.4f}")
            print(f"Best val loss: {summary['best_val_loss']:.4f}")
        print("=" * 60)

    @staticmethod
    def load(log_file: str) -> Dict[str, Any]:
        """Load experiment data from JSON file."""
        with open(log_file, 'r') as f:
            return json.load(f)


def plot_experiments(experiment_names: List[str], log_dir: str = "./logs",
                     output_file: str = "comparison.png"):
    """
    Plot and compare multiple experiments.

    Args:
        experiment_names: List of experiment names to compare
        log_dir: Directory containing log files
        output_file: Where to save the comparison plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    log_dir = Path(log_dir)
    experiments_data = []

    # Load all experiments
    for name in experiment_names:
        log_file = log_dir / f"{name}.json"
        if not log_file.exists():
            print(f"Warning: Log file not found for {name}")
            continue

        data = ExperimentLogger.load(log_file)
        experiments_data.append(data)

    if not experiments_data:
        print("No valid experiments found to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Experiment Comparison', fontsize=16)

    # Plot 1: Train Loss vs Iterations
    ax = axes[0, 0]
    for data in experiments_data:
        metrics = data['metrics']
        iterations = metrics['iterations']
        train_loss = [x for x in metrics['train_loss'] if x is not None]
        valid_iters = [iterations[i] for i, x in enumerate(metrics['train_loss']) if x is not None]
        ax.plot(valid_iters, train_loss, label=data['experiment_name'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss vs Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Train Loss vs Wall Time
    ax = axes[0, 1]
    for data in experiments_data:
        metrics = data['metrics']
        wall_time = [t / 3600 for t in metrics['wall_time']]  # Convert to hours
        train_loss = [x for x in metrics['train_loss'] if x is not None]
        valid_times = [wall_time[i] for i, x in enumerate(metrics['train_loss']) if x is not None]
        ax.plot(valid_times, train_loss, label=data['experiment_name'], linewidth=2)
    ax.set_xlabel('Wall Time (hours)')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss vs Wall Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation Loss vs Iterations
    ax = axes[1, 0]
    for data in experiments_data:
        metrics = data['metrics']
        iterations = metrics['iterations']
        val_loss = [x for x in metrics['val_loss'] if x is not None]
        valid_iters = [iterations[i] for i, x in enumerate(metrics['val_loss']) if x is not None]
        if valid_iters and val_loss:
            ax.plot(valid_iters, val_loss, label=data['experiment_name'],
                    linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning Rate Schedule
    ax = axes[1, 1]
    for data in experiments_data:
        metrics = data['metrics']
        iterations = metrics['iterations']
        lr = [x for x in metrics['learning_rate'] if x is not None]
        valid_iters = [iterations[i] for i, x in enumerate(metrics['learning_rate']) if x is not None]
        if valid_iters and lr:
            ax.plot(valid_iters, lr, label=data['experiment_name'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    plt.close()


def compare_experiments(experiment_names: List[str], log_dir: str = "./logs"):
    """
    Print a comparison table of multiple experiments.

    Args:
        experiment_names: List of experiment names to compare
        log_dir: Directory containing log files
    """
    log_dir = Path(log_dir)

    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)
    print(f"{'Experiment':<30} {'Iterations':<12} {'Time (h)':<10} {'Final Train':<12} {'Best Train':<12} {'Final Val':<12} {'Best Val':<12}")
    print("-" * 100)

    for name in experiment_names:
        log_file = log_dir / f"{name}.json"
        if not log_file.exists():
            print(f"{name:<30} [Log file not found]")
            continue

        data = ExperimentLogger.load(log_file)
        metrics = data['metrics']

        train_losses = [x for x in metrics['train_loss'] if x is not None]
        val_losses = [x for x in metrics['val_loss'] if x is not None]

        iterations = len(metrics['iterations'])
        time_hours = metrics['wall_time'][-1] / 3600 if metrics['wall_time'] else 0
        final_train = train_losses[-1] if train_losses else float('nan')
        best_train = min(train_losses) if train_losses else float('nan')
        final_val = val_losses[-1] if val_losses else float('nan')
        best_val = min(val_losses) if val_losses else float('nan')

        print(f"{name:<30} {iterations:<12} {time_hours:<10.2f} {final_train:<12.4f} {best_train:<12.4f} {final_val:<12.4f} {best_val:<12.4f}")

    print("=" * 100 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Example: Comparing experiments")

    # Simulate creating some experiment logs
    for i, name in enumerate(['baseline', 'larger_lr', 'more_layers']):
        logger = ExperimentLogger(name, config={'lr': 0.001 * (i + 1)})

        # Simulate training
        for step in range(100):
            train_loss = 5.0 * np.exp(-step / 50) + 0.5 + np.random.randn() * 0.1
            val_loss = 5.2 * np.exp(-step / 50) + 0.6 + np.random.randn() * 0.1 if step % 10 == 0 else None
            lr = 0.001 * (1 - step / 100)

            logger.log_step(
                iteration=step,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=lr,
                tokens_per_sec=1000 + np.random.randn() * 50
            )

        logger.save(force=True)
        logger.print_summary()

    # Compare experiments
    compare_experiments(['baseline', 'larger_lr', 'more_layers'])

    # Plot experiments
    plot_experiments(['baseline', 'larger_lr', 'more_layers'], output_file='example_comparison.png')
