"""
Training entry point.

Usage:
  python train.py                              # defaults from configs/default.yaml
  python train.py --n_nodes 20 --epochs 5     # quick test
  python train.py --resume checkpoints/best.pt
"""

import argparse
import os
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--mode", choices=["cvrp", "vrptw", "finetune"], default="cvrp",
                        help="cvrp: 3-feature nodes; vrptw: 5-feature; finetune: real-stats VRPTW")
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.n_nodes:
        config["training"]["n_nodes"] = args.n_nodes
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    device = args.device or get_device()

    from pomo.trainer import Trainer
    trainer = Trainer(config, device=device, mode=args.mode)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
