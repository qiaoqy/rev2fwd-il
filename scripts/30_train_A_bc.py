#!/usr/bin/env python3
"""Train Policy A with Behavior Cloning on the forward dataset.

=============================================================================
OVERVIEW
=============================================================================
This script implements Step 3 of the reverse-to-forward imitation learning
pipeline: training a neural network policy to perform the FORWARD pick-and-place
task using Behavior Cloning (BC) on the time-reversed dataset from Expert B.

The key idea is that Expert B's REVERSE trajectories (goal -> table), when
reversed in time, become valid FORWARD demonstrations (table -> goal). This
script trains an MLP policy to imitate these reversed demonstrations.

=============================================================================
BEHAVIOR CLONING ALGORITHM
=============================================================================
BC is a supervised learning approach to imitation learning:

    1. Dataset: {(obs_t, act_t)} pairs from expert demonstrations
    2. Policy: π_θ(obs) -> act (neural network with parameters θ)
    3. Loss: L(θ) = E[(π_θ(obs) - act_expert)²]  (MSE for continuous actions)
    4. Training: Minimize L(θ) using gradient descent (Adam optimizer)

For our 8-dim action space [ee_pose(7), gripper(1)], we use:
    - Position loss: MSE on XYZ coordinates
    - Quaternion loss: 1 - |dot(q_pred, q_gt)| (handles q ≡ -q equivalence)
    - Gripper loss: MSE (labels are -1/+1, predictions are tanh output)

=============================================================================
INPUT DATA FORMAT
=============================================================================
The input NPZ file (from script 20_make_A_forward_dataset.py) contains:
    - obs: (N, 36) float32 - Observations in FORWARD time order
    - act: (N, 8) float32  - Action labels [ee_pose(7), gripper(1)]
    - ep_id: (N,) int32    - Episode index for each sample

Where N is the total number of (obs, act) pairs across all episodes.

=============================================================================
OUTPUT FILES
=============================================================================
The script saves three files to the output directory (--out):

    1. model.pt - PyTorch checkpoint containing:
       - model_state_dict: Network weights
       - obs_dim: Observation dimension (36)
       - act_dim: Action dimension (8)
       - hidden: Tuple of hidden layer sizes

    2. norm.json - Observation normalization statistics:
       - mean: List of per-dimension means
       - std: List of per-dimension standard deviations

    3. config.yaml - Training configuration and results:
       - All hyperparameters (epochs, lr, batch_size, etc.)
       - Final loss values
       - Training time

=============================================================================
USAGE EXAMPLES
=============================================================================
# Standard training (recommended settings)
python scripts/30_train_A_bc.py \\
    --dataset data/A_forward_from_reverse.npz \\
    --out runs/bc_A \\
    --epochs 200 \\
    --batch_size 256 \\
    --lr 1e-3 \\
    --seed 0

# Quick test run
python scripts/30_train_A_bc.py \\
    --dataset data/A_forward_from_reverse.npz \\
    --out runs/bc_A_test \\
    --epochs 10 \\
    --batch_size 64

# Custom network architecture
python scripts/30_train_A_bc.py \\
    --dataset data/A_forward_from_reverse.npz \\
    --out runs/bc_A_large \\
    --hidden 512 512 256 \\
    --epochs 300

# Adjust loss weights (e.g., emphasize position accuracy)
python scripts/30_train_A_bc.py \\
    --dataset data/A_forward_from_reverse.npz \\
    --out runs/bc_A_pos \\
    --pos_weight 2.0 \\
    --quat_weight 1.0 \\
    --gripper_weight 0.5

=============================================================================
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace containing all parsed arguments with the following fields:
            - dataset: Path to input NPZ file
            - out: Output directory path
            - epochs: Number of training epochs
            - batch_size: Mini-batch size
            - lr: Learning rate for Adam optimizer
            - seed: Random seed for reproducibility
            - hidden: List of hidden layer sizes
            - pos_weight: Weight for position loss
            - quat_weight: Weight for quaternion loss
            - gripper_weight: Weight for gripper loss
            - print_every: Logging frequency (epochs)
            - device: Torch device (None for auto-detect)
    """
    parser = argparse.ArgumentParser(
        description="Train Policy A with Behavior Cloning on forward demonstrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =========================================================================
    # Input/Output Arguments
    # =========================================================================
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/A_forward_from_reverse.npz",
        help="Path to the forward BC dataset NPZ file. This file should contain "
             "'obs' and 'act' arrays from script 20_make_A_forward_dataset.py.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/bc_A",
        help="Output directory for saving model checkpoint (model.pt), "
             "normalization statistics (norm.json), and config (config.yaml).",
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs. Each epoch iterates over the entire "
             "dataset once. Default: 200.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for stochastic gradient descent. Larger batches "
             "provide more stable gradients but use more memory. Default: 256.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer. Default: 1e-3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Controls numpy, torch, and "
             "python random number generators. Default: 0.",
    )

    # =========================================================================
    # Model Architecture
    # =========================================================================
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes for the MLP policy. Specify as space-separated "
             "integers. Example: --hidden 256 256 (two layers of 256 units). "
             "Default: [256, 256].",
    )

    # =========================================================================
    # Loss Function Weights
    # =========================================================================
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Weight for position (XYZ) MSE loss. Increase to emphasize "
             "position accuracy. Default: 1.0.",
    )
    parser.add_argument(
        "--quat_weight",
        type=float,
        default=1.0,
        help="Weight for quaternion loss (1 - |dot(q_pred, q_gt)|). Controls "
             "importance of end-effector orientation. Default: 1.0.",
    )
    parser.add_argument(
        "--gripper_weight",
        type=float,
        default=1.0,
        help="Weight for gripper MSE loss. Controls importance of gripper "
             "open/close actions. Default: 1.0.",
    )

    # =========================================================================
    # Logging Options
    # =========================================================================
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print training loss every N epochs. Default: 10.",
    )

    # =========================================================================
    # Device Selection
    # =========================================================================
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for training ('cuda' or 'cpu'). If not specified, "
             "automatically selects CUDA if available, otherwise CPU.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for BC training.

    This function:
        1. Parses command-line arguments
        2. Imports the BC trainer module
        3. Runs the training loop
        4. Prints paths to saved outputs

    The actual training logic is implemented in rev2fwd_il.train.bc_trainer.train_bc().
    """
    args = _parse_args()

    # Import trainer (deferred to avoid slow imports before arg parsing)
    from rev2fwd_il.train.bc_trainer import train_bc

    # Run training
    result = train_bc(
        dataset_npz=args.dataset,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden=tuple(args.hidden),
        pos_weight=args.pos_weight,
        quat_weight=args.quat_weight,
        gripper_weight=args.gripper_weight,
        print_every=args.print_every,
        device=args.device,
    )

    print(f"\nTraining complete!")
    print(f"Model saved to: {result['model_path']}")
    print(f"Norm saved to: {result['norm_path']}")


if __name__ == "__main__":
    main()
